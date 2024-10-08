use std::thread;
use std::time::{Duration, Instant};

use crate::utils;
use crate::video::Frame;
use crossbeam::channel::{Receiver, RecvTimeoutError};
use ndarray::{s, Array4, Axis};
use ort::{inputs, OpenVINOExecutionProvider, Session, SessionOutputs};

#[derive(Clone, Debug)]
pub struct DetectConfig {
    pub device: String,
    pub model_path: String,
    pub target_size: usize,
    pub conf_thres: f32,
    pub iou_thres: f32,
    pub batch_size: usize,
    pub timeout: usize,
}

pub fn detect_worker(
    config: DetectConfig,
    array_q_recv: Receiver<Frame>,
) -> thread::JoinHandle<()> {
    let config = config.clone();
    thread::spawn(move || {
        let model = load_model(&config.model_path, &config.device).expect("Failed to load model");
        process_frames(array_q_recv, &model, &config);
    })
}

pub fn init_ort_runtime() -> anyhow::Result<()> {
    ort::init_from("lib/onnxruntime.dll").commit()?;
    anyhow::Ok(())
}

pub fn load_model(model_path: &str, device: &str) -> anyhow::Result<Session> {
    let model = Session::builder()?
        .with_execution_providers([OpenVINOExecutionProvider::default()
            .with_device_type(device)
            .build()
            .error_on_failure()])?
        .commit_from_file(model_path)?;
    anyhow::Ok(model)
}

pub fn process_frames(rx: Receiver<Frame>, model: &Session, config: &DetectConfig) {
    let mut frames: Vec<Frame> = Vec::new();
    let mut last_receive_time = Instant::now();
    let timeout = Duration::from_millis(config.timeout as u64);
    loop {
        if frames.len() >= config.batch_size || last_receive_time.elapsed() >= timeout {
            if !frames.is_empty() {
                // Process the batch of frames
                println!("Processing frame number: {}", frames.len());
                process_batch(&frames, model, config);
                frames.clear();
            }
            last_receive_time = Instant::now();
        }

        match rx.recv_timeout(timeout - last_receive_time.elapsed()) {
            Ok(frame_data) => {
                frames.push(frame_data);
                last_receive_time = Instant::now();
            }
            Err(RecvTimeoutError::Timeout) => {
                // Timeout occurred, process whatever frames we have
                if !frames.is_empty() {
                    println!("Timeout! Processing frame number: {}", frames.len());
                    process_batch(&frames, model, config);
                    frames.clear();
                }
                last_receive_time = Instant::now();
            }
            Err(RecvTimeoutError::Disconnected) => {
                if !frames.is_empty() {
                    process_batch(&frames, model, config);
                    println!("Disconnected! Processing frame number: {}", frames.len());
                    frames.clear();
                }
                // Channel disconnected, exit the loop
                break;
            }
        }
    }
}

pub fn process_batch(frames: &[Frame], model: &Session, config: &DetectConfig) {
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
        let mut boxes: Vec<utils::Bbox> = vec![];
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
            let bbox = utils::Bbox {
                class: class_id,
                score: prob,
                x1,
                y1,
                x2,
                y2,
            };
            boxes.push(bbox);
        }
        let nms_boxes = utils::nms(&mut boxes, true, 100, config.iou_thres);
        let video_path = frames[i].video_path.clone();
        let iframe_index = frames[i].iframe_index;
        for b in nms_boxes {
            println!(
                "Detected object in {:?} iframe {:?}: {:?} with confidence: {:?} at position: {:?}",
                video_path,
                iframe_index,
                b.class,
                b.score,
                (b.x1 as u32, b.y1 as u32, b.x2 as u32, b.y2 as u32)
            );
        }
    }
}

// pub fn process_frames(
//     rx: Receiver<Frame>,
//     model: &Session,
//     iou_threshold: f32,
//     conf_threshold: f32,
// ) {

//     while let Ok(frame_data) = rx.recv() {

//         let outputs: SessionOutputs = model
//             .run(inputs!["images" => frame_data.data.view()].unwrap())
//             .unwrap();

//         let output = outputs["output0"]
//             .try_extract_tensor::<f32>()
//             .unwrap()
//             .t()
//             .into_owned(); //[6, 102000, 1]

//         let output = output.slice(s![.., .., 0]); //[6, 102000]
//         let mut boxes: Vec<utils::Bbox> = vec![];

//         for row in output.axis_iter(Axis(1)) {
//             let row: Vec<_> = row.iter().copied().collect();
//             let class_id = row[5] as usize;
//             let prob = row[4];

//             if prob < conf_threshold {
//                 continue;
//             }

//             let bbox = utils::Bbox {
//                 class: class_id,
//                 score: prob,
//                 x1: row[0] as f32,
//                 y1: row[1] as f32,
//                 x2: row[2] as f32,
//                 y2: row[3] as f32,
//             };
//             boxes.push(bbox);
//         }

//         let nms_boxes = utils::nms(&mut boxes, true, 100, iou_threshold, conf_threshold);
//         for b in nms_boxes {
//             println!(
//                 "Detected object: {:?} with confidence: {:?} at position: {:?}",
//                 b.class,
//                 b.score,
//                 (b.x1, b.y1, b.x2, b.y2)
//             );
//         }
//     }
// }
