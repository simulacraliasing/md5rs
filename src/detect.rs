use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender};
use ndarray::{s, Array4, Axis};
use ort::{inputs, ExecutionProviderDispatch, Session, SessionOutputs};
use tracing::{debug, info, instrument, warn};

use crate::export::ExportFrame;
use crate::media::{ArrayItem, Frame};
use crate::utils::{nms, Bbox, Ep, EpDict};

#[derive(Clone, Debug)]
pub struct DetectConfig {
    pub device: String,
    pub model_path: PathBuf,
    pub target_size: usize,
    pub class_map: HashMap<usize, String>,
    pub conf_thres: f32,
    pub iou_thres: f32,
    pub batch_size: usize,
    pub timeout: usize,
    pub iframe: bool,
}

pub fn detect_worker(
    config: Arc<DetectConfig>,
    ep_dict: EpDict,
    array_q_recv: Receiver<ArrayItem>,
    export_q_s: Sender<ExportFrame>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut ep = ort::CPUExecutionProvider::default().build();
        for ep_info in ep_dict.eps {
            if ep_info.available {
                match ep_info.ep {
                    Ep::CoreML => {
                        ep = ort::CoreMLExecutionProvider::default()
                            .with_ane_only()
                            .with_subgraphs()
                            .build();
                    }
                    Ep::TensorRT => {
                        ep = ort::TensorRTExecutionProvider::default()
                            .with_engine_cache(true)
                            .with_engine_cache_path("./models")
                            .with_timing_cache(true)
                            .with_fp16(true)
                            .with_profile_min_shapes(format!(
                                "images:1x3x{}x{}",
                                config.target_size, config.target_size
                            ))
                            .with_profile_opt_shapes(format!(
                                "images:2x3x{}x{}",
                                config.target_size, config.target_size
                            ))
                            .with_profile_max_shapes(format!(
                                "images:5x3x{}x{}",
                                config.target_size, config.target_size
                            ))
                            .with_device_id(config.device.parse().unwrap_or(0))
                            .build();
                    }
                    Ep::CUDA => {
                        ep = ort::CUDAExecutionProvider::default()
                            .with_device_id(config.device.parse().unwrap_or(0))
                            .build();
                    }
                    Ep::OpenVINO => {
                        ep = ort::OpenVINOExecutionProvider::default()
                            .with_device_type(config.device.to_uppercase())
                            .build();
                    }
                    Ep::DirectML => {
                        ep = ort::DirectMLExecutionProvider::default()
                            .with_device_id(config.device.parse().unwrap_or(0))
                            .build();
                    }
                    Ep::Cpu => {
                        ep = ort::CPUExecutionProvider::default().build();
                    }
                }
                break;
            }
        }

        info!("Execution provider: {:?}", ep);

        let model = load_model(&config.model_path, ep).expect("Failed to load model");
        process_frames(array_q_recv, export_q_s, &model, &config).unwrap();
    })
}

pub fn load_model(model_path: &Path, ep: ExecutionProviderDispatch) -> Result<Session> {
    let model = Session::builder()?
        .with_execution_providers([ep])?
        .commit_from_file(model_path)?;

    Ok(model)
}

#[instrument]
pub fn process_frames(
    rx: Receiver<ArrayItem>,
    s: Sender<ExportFrame>,
    model: &Session,
    config: &DetectConfig,
) -> Result<()> {
    let mut frames: Vec<Frame> = Vec::new();
    let mut last_receive_time = Instant::now();
    let timeout = Duration::from_millis(config.timeout as u64);
    loop {
        if frames.len() >= config.batch_size || last_receive_time.elapsed() >= timeout {
            if !frames.is_empty() {
                // Process the batch of frames
                debug!("Processing frame number: {}", frames.len());
                process_batch(&frames, model, config, &s)?;
                frames.clear();
            }
            last_receive_time = Instant::now();
        }

        match rx.recv_timeout(timeout - last_receive_time.elapsed()) {
            Ok(item) => {
                match item {
                    ArrayItem::Frame(frame_data) => {
                        frames.push(frame_data);
                    }
                    ArrayItem::ErrFile(err_file) => s
                        .send(ExportFrame {
                            file: err_file.file,
                            shoot_time: None,
                            frame_index: 0,
                            total_frames: 1,
                            is_iframe: config.iframe,
                            bboxes: Some(vec![]),
                            label: None,
                            error: Some(err_file.error.to_string()),
                        })
                        .unwrap(),
                }

                last_receive_time = Instant::now();
            }
            Err(RecvTimeoutError::Timeout) => {
                // Timeout occurred, process whatever frames we have
                if !frames.is_empty() {
                    debug!(
                        "Recieve frame timeout! Processing frame number: {}",
                        frames.len()
                    );
                    process_batch(&frames, model, config, &s)?;
                    frames.clear();
                }
                last_receive_time = Instant::now();
            }
            Err(RecvTimeoutError::Disconnected) => {
                if !frames.is_empty() {
                    debug!(
                        "Channel disconnected! Processing frame number: {}",
                        frames.len()
                    );
                    process_batch(&frames, model, config, &s)?;
                    frames.clear();
                }
                // Channel disconnected, exit the loop
                break;
            }
        }
    }
    Ok(())
}

pub fn process_batch(
    frames: &[Frame],
    model: &Session,
    config: &DetectConfig,
    export_q_s: &Sender<ExportFrame>,
) -> Result<()> {
    let batch_size = frames.len();
    let mut inputs = Array4::<f32>::zeros((batch_size, 3, config.target_size, config.target_size));
    for (i, frame) in frames.iter().enumerate() {
        inputs
            .slice_mut(s![i, .., ..config.target_size, ..config.target_size])
            .assign(&frame.data);
    }
    let outputs: SessionOutputs = model
        .run(inputs!["images" => inputs.view()].unwrap())
        .unwrap();
    let output = outputs["output0"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .t()
        .into_owned(); //[6, 102000, batch]

    // Iterate batch/frame
    for i in 0..batch_size {
        let output = output.slice(s![.., .., i]); //[6, 102000]
        let mut boxes: Vec<Bbox> = vec![];
        // Iterate bboxes
        for row in output.axis_iter(Axis(1)) {
            let row: Vec<_> = row.iter().copied().collect();
            let class_id = row[5] as usize;
            let prob = row[4];
            if prob < config.conf_thres {
                continue;
            }
            let mut x1 = row[0] as f32 * frames[i].ratio - frames[i].padding.0 as f32;
            let mut y1 = row[1] as f32 * frames[i].ratio - frames[i].padding.1 as f32;
            let mut x2 = row[2] as f32 * frames[i].ratio - frames[i].padding.0 as f32;
            let mut y2 = row[3] as f32 * frames[i].ratio - frames[i].padding.1 as f32;
            x1 = x1.max(0.0).min(frames[i].width as f32);
            y1 = y1.max(0.0).min(frames[i].height as f32);
            x2 = x2.max(0.0).min(frames[i].width as f32);
            y2 = y2.max(0.0).min(frames[i].height as f32);
            let bbox = Bbox {
                class: class_id,
                score: prob,
                x1,
                y1,
                x2,
                y2,
            };
            boxes.push(bbox);
        }
        let nms_boxes = nms(&mut boxes, true, 100, config.iou_thres);

        let label = get_label(&nms_boxes, &config.class_map);

        let shoot_time = match frames[i].shoot_time {
            Some(shoot_time) => Some(shoot_time.to_string()),
            None => None,
        };

        let export_frame = ExportFrame {
            file: frames[i].file.clone(),
            shoot_time: shoot_time,
            frame_index: frames[i].iframe_index,
            total_frames: frames[i].total_frames,
            is_iframe: config.iframe,
            bboxes: Some(nms_boxes),
            label: Some(label),
            error: None,
        };
        export_q_s.send(export_frame).unwrap();
    }
    Ok(())
}

fn get_label(bboxes: &Vec<Bbox>, cls_map: &HashMap<usize, String>) -> HashSet<String> {
    let mut labels = HashSet::new();
    if bboxes.is_empty() {
        labels.insert("Blank".to_string());
        return labels;
    }

    for bbox in bboxes {
        let class_id = bbox.class;

        let label = match cls_map.get(&class_id) {
            Some(label) => label.to_string(),
            None => Err(anyhow!("Class ID not found")).unwrap(),
        };

        labels.insert(label);
    }
    labels
}
