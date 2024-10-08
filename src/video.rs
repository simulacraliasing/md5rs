use crate::utils::sample_evenly;
use anyhow::Ok;
use crossbeam::channel::Sender;
use ffmpeg_sidecar::child::FfmpegChild;
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel};
use ndarray::Array3;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct Frame {
    pub data: Array3<f32>,
    pub video_path: String,
    pub width: usize,
    pub height: usize,
    pub padding: (usize, usize),
    pub ratio: f32,
    pub iframe_index: usize,
}

pub fn media_worker(
    file_paths: Arc<Mutex<Vec<PathBuf>>>,
    imgsz: usize,
    max_frames: Option<usize>,
    array_q_s: Sender<Frame>,
) -> thread::JoinHandle<anyhow::Result<()>> {
    thread::spawn(move || -> anyhow::Result<()> {
        loop {
            let file_path = {
                let mut paths = file_paths.lock().unwrap();
                if paths.is_empty() {
                    break;
                }
                paths.pop()
            };

            if let Some(file_path) = file_path {
                if let Some(extension) = file_path.extension() {
                    let array_q_s = array_q_s.clone();
                    match extension.to_str().unwrap().to_lowercase().as_str() {
                        // "jpg" | "jpeg" | "png" | "gif" => process_image(&file_path)?,
                        "mp4" | "avi" | "mkv" | "mov" => {
                            process_video(&file_path, imgsz, max_frames, array_q_s)?
                        }
                        _ => (),
                    }
                }
            }
        }
        Ok(())
    })
}

pub fn process_video(
    video_path: &PathBuf,
    imgsz: usize,
    max_frames: Option<usize>,
    array_q_s: Sender<Frame>,
) -> anyhow::Result<()> {
    let video_path = video_path.to_string_lossy();
    let input = create_ffmpeg_command(&video_path, imgsz)?;

    handle_ffmpeg_output(input, array_q_s, imgsz, &video_path, max_frames)?;

    Ok(())
}

fn create_ffmpeg_command(video_path: &str, imgsz: usize) -> anyhow::Result<FfmpegChild> {
    let command = FfmpegCommand::new()
        .args(["-skip_frame", "nokey"])
        .input(video_path)
        .args(&[
            "-an",
            "-vf",
            &format!(
                "scale=w={}:h={}:force_original_aspect_ratio=decrease,pad={}:{}:(ow-iw)/2:(oh-ih)/2",
                imgsz, imgsz, imgsz, imgsz
            ),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-vsync",
            "vfr",
        ])
        .output("-")
        .spawn()?;
    Ok(command)
}

fn handle_ffmpeg_output(
    mut input: FfmpegChild,
    s: Sender<Frame>,
    imgsz: usize,
    video_path: &str,
    max_frames: Option<usize>,
) -> anyhow::Result<()> {
    let mut width = None;
    let mut height = None;
    let mut tmp_frames = vec![];
    input.iter()?.for_each(|e| match e {
        FfmpegEvent::Log(LogLevel::Error, e) => println!("Error: {}", e),
        FfmpegEvent::ParsedInputStream(i) => {
            if i.stream_type.to_lowercase() == "video" {
                width = Some(i.width as usize);
                height = Some(i.height as usize);
            }
        }
        FfmpegEvent::OutputFrame(f) => {
            tmp_frames.push(f);
        }
        _ => {}
    });
    let (sampled_frames, sampled_indexes) =
        sample_evenly(&tmp_frames, max_frames.unwrap_or(tmp_frames.len()));

    //calculate ratio and padding
    let width = width.expect("Failed to get video width");
    let height = height.expect("Failed to get video height");
    let pad = (width as i32 - height as i32).abs() / 2;
    let padding = if width > height {
        (0, pad as usize)
    } else {
        (pad as usize, 0)
    };
    let ratio = width.max(height) as f32 / imgsz as f32;

    for (f, i) in sampled_frames.into_iter().zip(sampled_indexes.into_iter()) {
        let ndarray_frame = Array3::from_shape_vec((imgsz, imgsz, 3), f.data).unwrap();
        let mut ndarray_frame = ndarray_frame.map(|&x| x as f32 / 255.0);
        ndarray_frame = ndarray_frame.permuted_axes([2, 0, 1]);
        // let ndarray_frame = ndarray_frame.insert_axis(Axis(0));
        let frame_data = Frame {
            data: ndarray_frame,
            video_path: video_path.to_string(),
            width,
            height,
            padding,
            ratio,
            iframe_index: i,
        };
        s.send(frame_data).expect("Send video frame failed");
    }
    Ok(())
}
