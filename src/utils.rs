use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use walkdir::{WalkDir, DirEntry};

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

pub fn sample_evenly<T: Clone>(list: &[T], sample_size: usize) -> (Vec<T>, Vec<usize>) {
    let len = list.len();
    if sample_size == 0 || len == 0 {
        return (Vec::new(), Vec::new());
    }

    let step = len as f64 / sample_size as f64;
    let mut sampled_elements = Vec::with_capacity(sample_size);
    let mut sampled_indexes = Vec::with_capacity(sample_size);
    for i in 0..sample_size {
        let index = (i as f64 * step).floor() as usize;
        sampled_elements.push(list[index].clone());
        sampled_indexes.push(index);
    }
    (sampled_elements, sampled_indexes)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash)]
pub struct FileItem {
    pub folder_id: usize,
    pub file_id: usize,
    pub file_path: PathBuf,
}

impl Eq for FileItem {}

fn is_label(entry: &DirEntry) -> bool {
    let skip_dirs = ["Animal", "Person", "Vehicle", "Blank"];
    entry
        .file_name()
        .to_str()
        .map(|s| skip_dirs.contains(&s))
        .unwrap_or(false)
}

pub fn index_files_and_folders(folder_path: &PathBuf) -> HashSet<FileItem> {
    let mut folder_id: usize = 0;
    let mut file_id: usize = 0;
    let mut file_paths = HashSet::new();
    

    for entry in  WalkDir::new(folder_path).sort_by_file_name().into_iter().filter_entry(|e| !is_label(e)) {
        let entry = entry.unwrap();
        if entry.file_type().is_dir() {
            folder_id += 1;
        } else if entry.file_type().is_file() {
            if is_video_photo(entry.path()) {
                file_paths.insert(FileItem {
                    folder_id,
                    file_id,
                    file_path: entry.path().to_path_buf(),
                });
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
