mod detect;
mod utils;
mod video;

use crate::detect::{detect_worker, init_ort_runtime, DetectConfig};
use crate::video::media_worker;
use crossbeam::channel::bounded;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use walkdir::WalkDir;

fn main() -> anyhow::Result<()> {
    let folder_path = "C:/Users/Zhengyi/git/Megatool/mock/测试";
    let detect_config = DetectConfig {
        device: String::from("GPU"),
        model_path: String::from("models/md_v5a_d_pp_fp16.onnx"),
        target_size: 1280,
        iou_thres: 0.45,
        conf_thres: 0.2,
        batch_size: 5,
        timeout: 10,
    };
    let imgsz = detect_config.target_size;
    let max_frames = Some(3);
    let start = Instant::now();

    let file_paths: Vec<PathBuf> = WalkDir::new(folder_path)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.path().to_path_buf())
        .collect();

    let file_paths = Arc::new(Mutex::new(file_paths));
    let mut media_handles = vec![];
    let mut detect_handles = vec![];

    let media_worker_threads = 12;

    let detect_worker_threads = 2;

    let (array_q_s, array_q_r) = bounded(6);

    init_ort_runtime().expect("Failed to initialize onnxruntime");
    for _ in 0..detect_worker_threads {
        let detect_config = detect_config.clone();
        let array_q_r = array_q_r.clone();
        let detect_handle = detect_worker(detect_config, array_q_r);
        detect_handles.push(detect_handle);
    }

    for _ in 0..media_worker_threads {
        let file_paths = Arc::clone(&file_paths);
        let array_q_s = array_q_s.clone();
        let media_handle = media_worker(file_paths, imgsz, max_frames, array_q_s);
        media_handles.push(media_handle);
    }

    for m_handle in media_handles {
        m_handle.join().unwrap();
    }

    drop(array_q_s);

    for d_handle in detect_handles {
        d_handle.join().unwrap();
    }

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    Ok(())
}
