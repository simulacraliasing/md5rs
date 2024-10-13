use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Ok, Result};
use clap::{Parser, ValueEnum};
use crossbeam_channel::{bounded, unbounded};
use rayon::prelude::*;
use tracing::{error, info, instrument, warn};

use export::ExportFrame;
use utils::FileItem;

mod detect;
mod export;
mod log;
mod media;
mod utils;

use crate::detect::{detect_worker, init_ort_runtime, DetectConfig};
use crate::export::{export, export_worker, parse_export_csv};
use crate::log::init_logger;
use crate::media::media_worker;
use crate::utils::index_files_and_folders;
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
    #[arg(short, long, value_enum, default_value_t = Device::Cpu)]
    device: Device,

    /// max frames to process per video
    #[arg(long)]
    max_frames: Option<usize>,

    /// decode only I frames in video
    #[arg(long, default_value_t = true)]
    iframe_only: bool,

    /// image size of model input
    #[arg(long, default_value_t = 1280)]
    imgsz: usize,

    /// batch size
    #[arg(short, long, default_value_t = 2)]
    batch: usize,

    /// number of detection worker threads
    #[arg(short, long, default_value_t = 2)]
    workers: usize,

    /// NMS IoU threshold
    #[arg(long, default_value_t = 0.45)]
    iou: f32,

    /// NMS confidence threshold
    #[arg(long, default_value_t = 0.2)]
    conf: f32,

    /// export format
    #[arg(long, value_enum, default_value_t = ExportFormat::Json)]
    export: ExportFormat,

    /// log level
    #[arg(long, default_value_t = String::from("info"))]
    log_level: String,

    /// log file
    #[arg(long, default_value_t = String::from("md5rs.log"))]
    log_file: String,

    /// checkpoint interval
    #[arg(long, default_value_t = 1)]
    checkpoint: usize,

    /// resume from checkpoint
    #[arg(long)]
    resume_from: Option<String>,
}

/// Enum for devices
#[derive(ValueEnum, Debug, Clone)]
#[value(rename_all = "kebab-case")]
enum Device {
    /// CPU device
    Cpu,

    /// GPU device
    Gpu,

    /// NPU device
    Npu,
}

/// Enum for export formats
#[derive(ValueEnum, Debug, Clone, Copy)]
#[value(rename_all = "kebab-case")]
enum ExportFormat {
    /// JSON format
    Json,

    /// CSV format
    Csv,
}

#[instrument]
fn main() -> Result<()> {

    let args: Args = Args::parse();

    let guard = init_logger(args.log_level, args.log_file).expect("Failed to initialize logger");

    if args.checkpoint == 0 {
        error!("Checkpoint should be greater than 0");
        return Ok(());
    }

    let folder_path = args.folder;
    let device = match args.device {
        Device::Cpu => "CPU",
        Device::Gpu => "GPU",
        Device::Npu => "NPU",
    };
    let detect_config = Arc::new(DetectConfig {
        device: device.to_string(),
        model_path: args.model,
        target_size: args.imgsz,
        iou_thres: args.iou,
        conf_thres: args.conf,
        batch_size: args.batch,
        timeout: 50,
        iframe: args.iframe_only,
    });
    let imgsz = args.imgsz;
    let max_frames = args.max_frames;
    let start = Instant::now();

    let mut file_paths = index_files_and_folders(&folder_path);

    let export_data = Arc::new(Mutex::new(Vec::new()));

    let file_paths = match args.resume_from {
        Some(checkpoint_path) => {
            let all_files =
                resume_from_checkpoint(&checkpoint_path, &mut file_paths, &export_data)?;
            all_files.to_owned()
        }
        None => file_paths,
    };

    let mut detect_handles = vec![];

    let mut export_handles = vec![];

    let (array_q_s, array_q_r) = bounded(args.batch * args.workers * 1);

    let (export_q_s, export_q_r) = unbounded();

    let checkpoint_counter = Arc::new(Mutex::new(0 as usize));

    //batch  0.6 2.2 3.2 7.4
    //thread 1.4 2.1 2.8
    init_ort_runtime().expect("Failed to initialize onnxruntime");
    for _ in 0..args.workers {
        let detect_config = Arc::clone(&detect_config);
        let array_q_r = array_q_r.clone();
        let export_q_s = export_q_s.clone();
        let detect_handle = detect_worker(detect_config, array_q_r, export_q_s);
        detect_handles.push(detect_handle);
    }

    for _ in 0..4 {
        let export_q_r = export_q_r.clone();
        let export_data = Arc::clone(&export_data);
        let folder_path = folder_path.clone();
        let checkpoint_counter = Arc::clone(&checkpoint_counter);
        let export_handle = std::thread::spawn(move || {
            export_worker(
                args.checkpoint,
                &checkpoint_counter,
                &args.export,
                &folder_path,
                export_q_r,
                &export_data,
            );
        });
        export_handles.push(export_handle);
    }

    file_paths.par_iter().for_each(|file| {
        let array_q_s = array_q_s.clone();
        media_worker(file.clone(), imgsz, args.iframe_only, max_frames, array_q_s);
    });

    drop(array_q_s);

    for d_handle in detect_handles {
        d_handle.join().unwrap();
    }

    drop(export_q_s);

    for e_handle in export_handles {
        e_handle.join().unwrap();
    }

    export(&folder_path, export_data, &args.export)?;

    let duration = start.elapsed();
    info!("Time elapsed: {:?}", duration);

    drop(guard);
    Ok(())
}

fn resume_from_checkpoint<'a>(
    checkpoint_path: &str,
    all_files: &'a mut HashSet<FileItem>,
    export_data: &Arc<Mutex<Vec<ExportFrame>>>,
) -> Result<&'a mut HashSet<FileItem>> {
    let checkpoint = Path::new(checkpoint_path);
    if !checkpoint.exists() {
        error!("Checkpoint file does not exist");
        return Err(anyhow::anyhow!("Checkpoint file does not exist"));
    }
    if !checkpoint.is_file() {
        error!("Checkpoint path is not a file");
        return Err(anyhow::anyhow!("Checkpoint path is not a file"));
    }
    match checkpoint.extension() {
        Some(ext) => {
            let ext = ext.to_str().unwrap();
            if ext != "json" && ext != "csv" {
                error!("Invalid checkpoint file extension: {}", ext);
                return Err(anyhow::anyhow!(
                    "Invalid checkpoint file extension: {}",
                    ext
                ));
            } else {
                let frames;
                if ext == "json" {
                    let json = std::fs::read_to_string(checkpoint)?;
                    frames = serde_json::from_str(&json)?;
                } else {
                    frames = parse_export_csv(checkpoint)?;
                }
                let mut file_frame_count = HashMap::new();
                let mut file_total_frames = HashMap::new();
                for f in &frames {
                    let file = &f.file;
                    let count = file_frame_count.entry(file.clone()).or_insert(0);
                    *count += 1;
                    file_total_frames
                        .entry(file.clone())
                        .or_insert(f.total_frames);

                    if let Some(total_frames) = file_total_frames.get(&file) {
                        if let Some(frame_count) = file_frame_count.get(&file) {
                            if total_frames == frame_count {
                                all_files.remove(&file);
                            }
                        }
                    }
                }
                export_data.lock().unwrap().extend_from_slice(&frames);
                Ok(all_files)
            }
        }
        None => {
            error!("Invalid checkpoint file extension");
            return Err(anyhow::anyhow!("Invalid checkpoint file extension"));
        }
    }
}
