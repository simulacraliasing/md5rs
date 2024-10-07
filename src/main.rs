use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel};
use ndarray::{s, Array3, Array4, Axis};
use ort::{inputs, GraphOptimizationLevel, OpenVINOExecutionProvider, Session, SessionOutputs};
use std::sync::mpsc::channel;
use std::thread;
use std::time::{Duration, Instant};
mod utils;

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let video_path = "input/IMG_0251.mp4";
    let target_size = 640;

    let mut input = FfmpegCommand::new()
        .args(["-skip_frame", "nokey"])
        .input(video_path)
        .args(&[
            "-vf",
            &format!(
        "scale=w={}:h={}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2",
        target_size, target_size, target_size, target_size
        ),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "vfr",
        ])
        // .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "vfr"])
        .output("-") // <- Discoverable aliases for FFmpeg args // <- Convenient argument presets
        .spawn()
        .unwrap(); // <- Uses an ordinary `std::process::Child`

    let (tx, rx) = channel::<(Array4<f32>, String)>();

    let work_handle = std::thread::spawn(move || {
        ort::init_from("lib/onnxruntime.dll").commit().unwrap();

        let model = Session::builder()
            .unwrap()
            .with_execution_providers([OpenVINOExecutionProvider::default()
                .with_device_type("CPU")
                .build()
                .error_on_failure()])
            .unwrap()
            .commit_from_file("yolov8n.onnx")
            // .commit_from_file("C:\\Users\\Zhengyi\\git\\Megatool\\models\\md_v5a.0.0.onnx")
            .unwrap();

        let mut frame_count = 0;

        while let Ok((input, video_path)) = rx.recv() {
            frame_count += 1;
            println!("Processing frame: {}", frame_count);
            let outputs: SessionOutputs = model
                .run(inputs!["images" => input.view()].unwrap())
                .unwrap();
            let output = outputs["output0"]
                .try_extract_tensor::<f32>()
                .unwrap()
                .t()
                .into_owned();
            let output = output.slice(s![.., .., 0]);
            let mut boxes: Vec<utils::Bbox> = vec![];
            for row in output.axis_iter(Axis(0)) {
                let row: Vec<_> = row.iter().copied().collect();
                let (class_id, prob) = row
                    .iter()
                    // skip bounding box coordinates
                    .skip(4)
                    .enumerate()
                    .map(|(index, value)| (index, *value))
                    .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                    .unwrap();
                if prob < 0.1 {
                    continue;
                }
                let label = class_id;
                let xc = row[0];
                let yc = row[1];
                let w = row[2];
                let h = row[3];
                let bbox = utils::Bbox {
                    class: label,
                    score: prob,
                    x1: (xc - w / 2.0) as f32,
                    y1: (yc - h / 2.0) as f32,
                    x2: (xc + w / 2.0) as f32,
                    y2: (yc + h / 2.0) as f32,
                };
                boxes.push(bbox);
                // println!(
                //     "Detected object: {:?} with confidence: {:?} at position: {:?}",
                //     label,
                //     prob,
                //     (xc, yc, w, h)
                // );
            }
            let nms_boxes = utils::nms(&mut boxes, true, 100, 0.4, 0.2);
            for b in nms_boxes {
                println!(
                    "Detected object: {:?} with confidence: {:?} at position: {:?}",
                    b.class,
                    b.score,
                    (b.x1, b.y1, b.x2, b.y2)
                );
            }
        }
    });

    input.iter().unwrap().for_each(|e| match e {
        FfmpegEvent::Log(LogLevel::Error, e) => println!("Error: {}", e),
        FfmpegEvent::Progress(p) => println!("Progress: {}", p.time),
        FfmpegEvent::OutputFrame(f) => {
            let ndarray_frame =
                Array3::from_shape_vec((target_size, target_size, 3), f.data).unwrap();
            let mut ndarray_frame = ndarray_frame.map(|&x| x as f32 / 255.0);
            ndarray_frame = ndarray_frame.permuted_axes([2, 0, 1]);
            let ndarray_frame = ndarray_frame.insert_axis(Axis(0));
            tx.send((ndarray_frame, video_path.to_string())).unwrap();
        }
        _ => {}
    });

    drop(tx);
    work_handle.join().unwrap();

    let duration = start.elapsed();
    println!("Time elapsed: {:?}", duration);
    Ok(())
}
