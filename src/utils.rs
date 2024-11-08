use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ort::{ExecutionProvider, Session};
use serde::{Deserialize, Serialize};
use tracing::info;
use walkdir::{DirEntry, WalkDir};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bbox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    pub class: usize,
}

impl Bbox {
    fn area(&self) -> f32 {
        (self.x2 - self.x1) * (self.y2 - self.y1)
    }
}

fn iou(box1: &Bbox, box2: &Bbox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    let intersection_area = ((x2 - x1).max(0.0)) * ((y2 - y1).max(0.0));
    let union_area = box1.area() + box2.area() - intersection_area;

    if union_area == 0.0 {
        0.0
    } else {
        intersection_area / union_area
    }
}

pub fn nms(boxes: &mut Vec<Bbox>, agnostic: bool, topk: usize, iou_threshold: f32) -> Vec<Bbox> {
    // Sort boxes by score in descending order
    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut result = Vec::new();

    if agnostic {
        // Perform agnostic NMS
        while !boxes.is_empty() {
            let best_box = boxes.remove(0);
            result.push(best_box.clone());

            if result.len() >= topk {
                break;
            }

            boxes.retain(|b| iou(&best_box, b) < iou_threshold);
        }
    } else {
        // Perform class-specific NMS
        let mut class_map: std::collections::HashMap<usize, Vec<Bbox>> =
            std::collections::HashMap::new();

        for b in boxes.clone() {
            class_map.entry(b.class).or_insert_with(Vec::new).push(b);
        }

        for (_, mut class_boxes) in class_map {
            while !class_boxes.is_empty() {
                let best_box = class_boxes.remove(0);
                result.push(best_box.clone());

                if result.iter().filter(|b| b.class == best_box.class).count() >= topk {
                    break;
                }

                class_boxes.retain(|b| iou(&best_box, b) < iou_threshold);
            }
        }
    }

    result
}

pub fn sample_evenly<T: Clone>(list: &[T], sample_size: usize) -> Vec<T> {
    let len = list.len();
    if sample_size == 0 || len == 0 {
        return Vec::new();
    }

    let step = len as f64 / sample_size as f64;
    let mut sampled_elements = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        let index = (i as f64 * step).floor() as usize;
        sampled_elements.push(list[index].clone());
    }
    sampled_elements
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub struct FileItem {
    pub folder_id: usize,
    pub file_id: usize,
    pub file_path: PathBuf,
    #[serde(skip_serializing)]
    pub tmp_path: PathBuf,
}

impl Eq for FileItem {}

impl FileItem {
    pub fn new(
        folder_id: usize,
        file_id: usize,
        file_path: PathBuf,
        tmp_path: Option<PathBuf>,
    ) -> Self {
        match tmp_path {
            Some(tmp_path) => Self {
                folder_id,
                file_id,
                file_path,
                tmp_path: tmp_path,
            },
            None => Self {
                folder_id,
                file_id,
                file_path: file_path.clone(),
                tmp_path: file_path,
            },
        }
    }
}

fn is_skip(entry: &DirEntry) -> bool {
    let skip_dirs = ["Animal", "Person", "Vehicle", "Blank"];
    entry
        .file_name()
        .to_str()
        .map(|s| {
            skip_dirs.contains(&s) || s.starts_with('.') || s == "result.csv" || s == "result.json"
        })
        .unwrap_or(false)
}

pub fn index_files_and_folders(folder_path: &PathBuf) -> HashSet<FileItem> {
    let mut folder_id: usize = 0;
    let mut file_id: usize = 0;
    let mut file_paths = HashSet::new();

    for entry in WalkDir::new(folder_path)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|e| !is_skip(e))
    {
        let entry = entry.unwrap();
        if entry.file_type().is_dir() {
            folder_id += 1;
        } else if entry.file_type().is_file() {
            if is_video_photo(entry.path()) {
                file_paths.insert(FileItem::new(
                    folder_id,
                    file_id,
                    entry.path().to_path_buf(),
                    None,
                ));
                file_id += 1;
            }
        }
    }

    file_paths
}

fn is_video_photo(path: &Path) -> bool {
    if let Some(extension) = path.extension() {
        match extension.to_str().unwrap().to_lowercase().as_str() {
            "mp4" | "avi" | "mkv" | "mov" => true,
            "jpg" | "jpeg" | "png" => true,
            _ => false,
        }
    } else {
        false
    }
}

// EP availability check

#[derive(Serialize, Deserialize, Clone)]
pub enum Ep {
    CoreML,
    TensorRT,
    CUDA,
    OpenVINO,
    DirectML,
    Cpu,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EpInfo {
    pub ep: Ep,
    pub available: bool,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct EpDict {
    pub device: String,
    pub eps: Vec<EpInfo>,
}

impl EpDict {
    pub fn save(self) -> Result<()> {
        // save dict to json
        let json = serde_json::to_string_pretty(&self)?;
        let json_file_name = std::format!("epinfo_{}.json", self.device);
        let json_path = Path::new(&json_file_name);
        let mut file = File::create(json_path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

fn check_ep_availability(device: &str) -> Result<()> {
    info!("Checking available execution providers");

    let mut ep_infos = Vec::new();

    #[cfg(target_os = "macos")]
    {
        let coreml = ort::CoreMLExecutionProvider::default();
        if coreml.is_available().unwrap() {
            match Session::builder()?
                .with_execution_providers(vec![coreml.build().error_on_failure()])
            {
                Ok(_) => {
                    let ep_info = EpInfo {
                        ep: Ep::CoreML,
                        available: true,
                    };
                    ep_infos.push(ep_info);
                }
                Err(_e) => {
                    let ep_info = EpInfo {
                        ep: Ep::CoreML,
                        available: false,
                    };
                    ep_infos.push(ep_info);
                }
            }
        } else {
            let ep_info = EpInfo {
                ep: Ep::CoreML,
                available: false,
            };
            ep_infos.push(ep_info);
        }
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        let tensor_rt =
            ort::TensorRTExecutionProvider::default().with_device_id(device.parse().unwrap_or(0));
        if tensor_rt.is_available().unwrap() {
            match Session::builder()?
                .with_execution_providers(vec![tensor_rt.build().error_on_failure()])
            {
                Ok(_) => {
                    let ep_info = EpInfo {
                        ep: Ep::TensorRT,
                        available: true,
                    };
                    ep_infos.push(ep_info);
                }
                Err(_e) => {
                    let ep_info = EpInfo {
                        ep: Ep::TensorRT,
                        available: false,
                    };
                    ep_infos.push(ep_info);
                }
            }
        } else {
            let ep_info = EpInfo {
                ep: Ep::TensorRT,
                available: false,
            };
            ep_infos.push(ep_info);
        }

        let cuda =
            ort::CUDAExecutionProvider::default().with_device_id(device.parse().unwrap_or(0));
        if cuda.is_available().unwrap() {
            match Session::builder()?
                .with_execution_providers(vec![cuda.build().error_on_failure()])
            {
                Ok(_) => {
                    let ep_info = EpInfo {
                        ep: Ep::CUDA,
                        available: true,
                    };
                    ep_infos.push(ep_info);
                }
                Err(_e) => {
                    let ep_info = EpInfo {
                        ep: Ep::CUDA,
                        available: false,
                    };
                    ep_infos.push(ep_info);
                }
            }
        } else {
            let ep_info = EpInfo {
                ep: Ep::CUDA,
                available: false,
            };
            ep_infos.push(ep_info);
        }

        let open_vino =
            ort::OpenVINOExecutionProvider::default().with_device_type(device.to_uppercase());
        if open_vino.is_available().unwrap() {
            match Session::builder()?
                .with_execution_providers(vec![open_vino.build().error_on_failure()])
            {
                Ok(_) => {
                    let ep_info = EpInfo {
                        ep: Ep::OpenVINO,
                        available: true,
                    };
                    ep_infos.push(ep_info);
                }
                Err(_e) => {
                    let ep_info = EpInfo {
                        ep: Ep::OpenVINO,
                        available: false,
                    };
                    ep_infos.push(ep_info);
                }
            }
        } else {
            let ep_info = EpInfo {
                ep: Ep::OpenVINO,
                available: false,
            };
            ep_infos.push(ep_info);
        }
    }

    #[cfg(target_os = "windows")]
    {
        let dml =
            ort::DirectMLExecutionProvider::default().with_device_id(device.parse().unwrap_or(0));
        if dml.is_available().unwrap() {
            match Session::builder()?.with_execution_providers(vec![dml.build().error_on_failure()])
            {
                Ok(_) => {
                    let ep_info = EpInfo {
                        ep: Ep::DirectML,
                        available: true,
                    };
                    ep_infos.push(ep_info);
                }
                Err(_e) => {
                    let ep_info = EpInfo {
                        ep: Ep::DirectML,
                        available: false,
                    };
                    ep_infos.push(ep_info);
                }
            }
        } else {
            let ep_info = EpInfo {
                ep: Ep::DirectML,
                available: false,
            };
            ep_infos.push(ep_info);
        }
    }

    let ep_dict = EpDict {
        device: device.to_string(),
        eps: ep_infos,
    };

    ep_dict.save()?;

    Ok(())
}

pub fn read_ep_dict(device: &str) -> Result<EpDict> {
    if device == "cpu" {
        return Ok(EpDict {
            device: device.to_string(),
            eps: vec![EpInfo {
                ep: Ep::Cpu,
                available: true,
            }],
        });
    }
    let json_file_name = std::format!("epinfo_{}.json", device);
    let json_path = Path::new(&json_file_name);
    let ep_dict: EpDict;
    if json_path.exists() {
        let json = std::fs::read_to_string(json_path)?;
        ep_dict = serde_json::from_str(&json)?;
    } else {
        check_ep_availability(device)?;
        ep_dict = read_ep_dict(device)?;
    }
    Ok(ep_dict)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub path: PathBuf,
    pub imgsz: usize,
    pub classes: BTreeSet<String>,
}

impl PartialEq for ModelConfig {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.path == other.path
            && self.imgsz == other.imgsz
            && self.classes == other.classes
    }
}

impl Eq for ModelConfig {}

#[allow(dead_code)]
impl ModelConfig {
    pub fn save<P: AsRef<Path>>(&self, toml: P) -> Result<()> {
        let toml_str = toml::to_string_pretty(&self)?;
        let mut file = File::create(toml)?;
        file.write_all(toml_str.as_bytes())?;
        Ok(())
    }

    pub fn class_map(&self) -> HashMap<usize, String> {
        let mut class_map = HashMap::new();
        for (i, class) in self.classes.iter().enumerate() {
            class_map.insert(i, class.clone());
        }
        class_map
    }
}

pub fn load_model_config<P: AsRef<Path>>(config: P) -> Result<ModelConfig> {
    let toml_str = std::fs::read_to_string(config)?;
    let model_config: ModelConfig = toml::from_str(&toml_str)?;
    Ok(model_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_save() {
        let model = ModelConfig {
            name: "mdv5a".to_string(),
            path: PathBuf::from("models/md_v5a_d_pp.onnx"),
            imgsz: 1280,
            classes: BTreeSet::from([
                "Animal".to_string(),
                "Person".to_string(),
                "Vehicle".to_string(),
            ]),
        };
        let toml_path = "models/md5va.toml";
        model.save(toml_path).unwrap();
        let target = load_model_config(toml_path).unwrap();
        assert_eq!(model, target);
    }

    #[test]
    fn test_model_config_class_map() {
        let model = ModelConfig {
            name: "mdv5a".to_string(),
            path: PathBuf::from("models/md_v5a_d_pp.onnx"),
            imgsz: 1280,
            classes: BTreeSet::from([
                "Animal".to_string(),
                "Person".to_string(),
                "Vehicle".to_string(),
            ]),
        };
        let target = HashMap::from([
            (0, "Animal".to_string()),
            (1, "Person".to_string()),
            (2, "Vehicle".to_string()),
        ]);
        assert_eq!(model.class_map(), target);
    }

    #[test]
    fn test_load_model_config() {
        let model = load_model_config("models/mdv5a.toml").unwrap();
        let target = ModelConfig {
            name: "mdv5a".to_string(),
            path: PathBuf::from("models/md_v5a_d_pp.onnx"),
            imgsz: 1280,
            classes: BTreeSet::from([
                "Animal".to_string(),
                "Person".to_string(),
                "Vehicle".to_string(),
            ]),
        };
        assert_eq!(model, target);
    }
}
