mod detect;
mod export;
mod media;
mod utils;

use crate::detect::{detect_worker, init_ort_runtime, DetectConfig};
use crate::media::media_worker;
use crate::utils::index_files_and_folders;
use clap::{Parser, ValueEnum};
use crossbeam_channel::{bounded, unbounded};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
    #[arg(long, default_value_t = 2)]
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

fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();
    let folder_path = args.folder;
    let device = match args.device {
        Device::Cpu => "CPU",
        Device::Gpu => "GPU",
        Device::Npu => "NPU",
    };
    let detect_config = DetectConfig {
        device: device.to_string(),
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

    let file_paths = index_files_and_folders(folder_path);

    let mut detect_handles = vec![];

    let mut export_handles = vec![];

    let (array_q_s, array_q_r) = bounded(args.batch * args.workers * 1);

    let (export_q_s, export_q_r) = unbounded();

    let export_data = Arc::new(Mutex::new(Vec::new()));

    //batch  0.6 2.2 3.2 7.4
    //thread 1.4 2.1 2.8
    init_ort_runtime().expect("Failed to initialize onnxruntime");
    for _ in 0..args.workers {
        let detect_config = detect_config.clone();
        let array_q_r = array_q_r.clone();
        let export_q_s = export_q_s.clone();
        let detect_handle = detect_worker(detect_config, array_q_r, export_q_s);
        detect_handles.push(detect_handle);
    }

    for _ in 0..4 {
        let export_q_r = export_q_r.clone();
        let export_data = Arc::clone(&export_data);
        let export_handle = std::thread::spawn(move || {
            export::export_worker(export_q_r, &export_data);
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

    match args.export {
        ExportFormat::Json => {
            let export_data = Arc::try_unwrap(export_data).unwrap().into_inner().unwrap();
            let json = serde_json::to_string_pretty(&export_data).unwrap();
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .open("output/output.json")
                .unwrap();
            file.write_all(json.as_bytes()).unwrap();
        }
        ExportFormat::Csv => {
            let export_data = Arc::try_unwrap(export_data).unwrap().into_inner().unwrap();
            let csv = export_data
                .iter()
                .map(|export_frame| {
                    format!(
                        "{},{},{},{},{},{},{},{}",
                        export_frame.file.folder_id,
                        export_frame.file.file_id,
                        export_frame.file.file_path.to_string_lossy(),
                        export_frame.frame_index,
                        export_frame.is_iframe,
                        format!(
                            "\"{}\"",
                            serde_json::to_string(&export_frame.bboxes)
                                .unwrap()
                                .replace("\"", "\"\"")
                        ),
                        export_frame.label,
                        export_frame.error.clone().unwrap_or("null".to_string())
                    )
                })
                .collect::<Vec<String>>()
                .join("\n");
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .open("output/output.csv")
                .unwrap();
            file.write_all(
                "folder_id,file_id,file_path,frame_index,is_iframe,bboxes,label,error\n".as_bytes(),
            )
            .unwrap();
            file.write_all(csv.as_bytes()).unwrap();
        }
    }

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    Ok(())
}
