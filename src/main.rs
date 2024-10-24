use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use crossbeam_channel::{bounded, unbounded};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use tracing::{error, info, instrument, warn};

use export::ExportFrame;
use utils::{load_model_config, read_ep_dict, FileItem};

mod detect;
mod export;
mod io;
mod log;
mod media;
mod utils;

use crate::detect::{detect_worker, DetectConfig};
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

    /// path to the model toml file
    #[arg(short, long, default_value = "models/md_v5a_fp16.toml")]
    model: String,

    /// device to run the model.
    /// Available options: cpu|gpu|npu for openvino;
    /// 0|1|2.. for cuda, tensorrt, directml
    #[arg(short, long, default_value = "cpu")]
    device: Vec<String>,

    /// max frames to process per video. Set to None to process all frames
    #[arg(long, default_value = "3")]
    max_frames: Option<usize>,

    /// decode only I frames in video.
    /// In short, it helps decode video faster by skip harder frames.
    /// Check https://en.wikipedia.org/wiki/Video_compression_picture_types to understand I frames
    #[arg(long, short, default_value_t = true)]
    iframe_only: bool,

    /// batch size. Batch size will increase
    #[arg(short, long, default_value_t = 2)]
    batch: usize,

    /// number of detection worker threads
    #[arg(short, long, default_value = "2")]
    workers: Vec<usize>,

    /// NMS IoU threshold
    #[arg(long, default_value_t = 0.45)]
    iou: f32,

    /// NMS confidence threshold
    #[arg(long, default_value_t = 0.2)]
    conf: f32,

    /// export format
    #[arg(short, long, value_enum, default_value_t = ExportFormat::Json)]
    export: ExportFormat,

    /// log level
    #[arg(long, default_value = "info")]
    log_level: String,

    /// log file
    #[arg(long, default_value = "md5rs.log")]
    log_file: String,

    /// checkpoint interval.
    /// Will export data to disk every N frames(not files!).
    /// Set it too low could affect performance
    #[arg(long, default_value_t = 100)]
    checkpoint: usize,

    /// resume from checkpoint file(the same path as export file unless you renamed it)
    #[arg(long)]
    resume_from: Option<String>,

    /// SSD buffer path. Could help if speed is IO bound when data stored in HDD(make sure it is a SSD path!)
    #[arg(long)]
    buffer_path: Option<String>,

    /// buffer size. Max files to keep in buffer, adjust on SSD free space
    #[arg(long, default_value_t = 20)]
    buffer_size: usize,
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

    let buffer_path = args.buffer_path.clone();

    info!("Cleaning up buffer");
    match cleanup_buffer(&buffer_path) {
        Ok(_) => {}
        Err(e) => {
            error!("Error cleaning up buffer: {:?}", e);
        }
    }

    if args.checkpoint == 0 {
        error!("Checkpoint should be greater than 0");
        return Ok(());
    }

    let folder_path = std::path::PathBuf::from(&args.folder);
    let folder_path = std::fs::canonicalize(folder_path).expect("Folder doesn't exist");

    let model_config = load_model_config(&args.model).expect("Failed to load model config");

    let imgsz = model_config.imgsz;
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

    let (array_q_s, array_q_r) = bounded(args.batch * args.workers.iter().sum::<usize>() * 2);

    let (export_q_s, export_q_r) = unbounded();

    let checkpoint_counter = Arc::new(Mutex::new(0 as usize));

    for (i, d) in args.device.iter().enumerate() {
        let detect_config = Arc::new(DetectConfig {
            device: d.clone(),
            model_path: model_config.path.clone(),
            target_size: model_config.imgsz,
            class_map: model_config.class_map(),
            iou_thres: args.iou,
            conf_thres: args.conf,
            batch_size: args.batch,
            timeout: 50,
            iframe: args.iframe_only,
        });
        let ep_dict = read_ep_dict(&d)?;
        for _ in 0..args.workers[i] {
            let detect_config = Arc::clone(&detect_config);
            let array_q_r = array_q_r.clone();
            let export_q_s = export_q_s.clone();
            let ep_dict = ep_dict.clone();
            let detect_handle = detect_worker(detect_config, ep_dict, array_q_r, export_q_s);
            detect_handles.push(detect_handle);
        }
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

    let pb = ProgressBar::new(file_paths.len() as u64);

    pb.set_style(ProgressStyle::default_bar().template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
    )?);

    let (io_q_s, io_q_r) = bounded(args.buffer_size);

    match &args.buffer_path {
        Some(buffer_path) => {
            let buffer_path = std::path::PathBuf::from(buffer_path);
            std::fs::create_dir_all(&buffer_path)?;
            let buffer_path = std::fs::canonicalize(buffer_path)?;

            let io_handle = std::thread::spawn(move || {
                for file in file_paths.iter() {
                    io::io_worker(&buffer_path, file, io_q_s.clone()).unwrap();
                }
                drop(io_q_s);
            });

            io_q_r
                .iter()
                .par_bridge()
                .progress_with(pb.clone())
                .for_each(|file| {
                    let array_q_s = array_q_s.clone();
                    media_worker(file, imgsz, args.iframe_only, max_frames, array_q_s);
                });
            io_handle.join().unwrap();
        }
        None => {
            file_paths
                .par_iter()
                .progress_with(pb.clone())
                .for_each(|file| {
                    let array_q_s = array_q_s.clone();
                    media_worker(file.clone(), imgsz, args.iframe_only, max_frames, array_q_s);
                });
        }
    }

    drop(array_q_s);

    for d_handle in detect_handles {
        match d_handle.join() {
            Ok(_) => {}
            Err(e) => {
                error!("Error joining detect worker: {:?}", e);
                std::process::exit(1);
            }
        }
    }

    drop(export_q_s);

    for e_handle in export_handles {
        match e_handle.join() {
            Ok(_) => {}
            Err(e) => {
                error!("Error joining export worker: {:?}", e);
                std::process::exit(1);
            }
        }
    }

    export(&folder_path, export_data, &args.export)?;

    let duration = start.elapsed();
    info!("Time elapsed: {:?}", duration);
    pb.finish_and_clear();

    cleanup_buffer(&args.buffer_path)?;

    drop(guard);
    Ok(())
}

fn cleanup_buffer(buffer_path: &Option<String>) -> Result<()> {
    if let Some(buff_path) = buffer_path {
        let buff_path = std::path::PathBuf::from(buff_path);
        if buff_path.exists() {
            std::fs::remove_dir_all(&buff_path)?;
        }
    }
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
