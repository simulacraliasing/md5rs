mod detect;
mod utils;
mod media;

use crate::detect::{detect_worker, init_ort_runtime, DetectConfig};
use crate::media::media_worker;
use clap::Parser;
use crossbeam_channel::bounded;
use rayon::prelude::*;
use std::path::PathBuf;
use std::time::Instant;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// folder to process
    #[arg(short, long)]
    folder: String,

    /// path to the model
    #[arg(short, long, default_value_t = String::from("models/md_v5a_d_pp_fp16.onnx"))]
    model: String,

    /// device to run the model
    #[arg(short, long, default_value_t = String::from("CPU"))]
    device: String,

    /// max frames to process per video
    #[arg(long)]
    max_frames: Option<usize>,

    /// image size of model input
    #[arg(long, default_value_t = 1280)]
    imgsz: usize,

    /// batch size
    #[arg(short, long, default_value_t = 2)]
    batch: usize,

    /// number of detection worker threads
    #[arg(long, default_value_t = 2)]
    workers: usize,

    
    /// NMS IoU threshold
    #[arg(long, default_value_t = 0.45)]
    iou: f32,

    /// NMS confidence threshold
    #[arg(long, default_value_t = 0.2)]
    conf: f32,
}

fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();
    let folder_path = args.folder;
    let detect_config = DetectConfig {
        device: args.device,
        model_path: args.model,
        target_size: args.imgsz,
        iou_thres: args.iou,
        conf_thres: args.conf,
        batch_size: args.batch,
        timeout: 10,
    };
    let imgsz = args.imgsz;
    let max_frames = args.max_frames;
    let start = Instant::now();

    let file_paths: Vec<PathBuf> = WalkDir::new(folder_path)
        .into_iter()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.path().to_path_buf())
        .collect();

    let mut detect_handles = vec![];

    let detect_worker_threads = args.workers;

    let (array_q_s, array_q_r) = bounded(args.batch * args.workers * 1);
    //batch  0.6 2.2 3.2 7.4
    //thread 1.4 2.1 2.8
    init_ort_runtime().expect("Failed to initialize onnxruntime");
    for _ in 0..detect_worker_threads {
        let detect_config = detect_config.clone();
        let array_q_r = array_q_r.clone();
        let detect_handle = detect_worker(detect_config, array_q_r);
        detect_handles.push(detect_handle);
    }

    file_paths.par_iter().for_each(|file_path| {
        let array_q_s = array_q_s.clone();

        media_worker(file_path.clone(), imgsz, max_frames, array_q_s);
    });

    drop(array_q_s);

    for d_handle in detect_handles {
        d_handle.join().unwrap();
    }

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    Ok(())
}
