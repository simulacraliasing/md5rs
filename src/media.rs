use crate::utils::{sample_evenly, FileItem};

use anyhow::{Context, Result};
use chrono::{DateTime, Local};
use crossbeam_channel::Sender;
use fast_image_resize::{ResizeAlg, ResizeOptions, Resizer};
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel};
use ffmpeg_sidecar::iter::FfmpegIterator;
use image::{DynamicImage, GenericImageView, ImageReader};
use jpeg_decoder::Decoder;
use ndarray::{s, Array3, Dim};
use nom_exif::{Exif, ExifIter, ExifTag, MediaParser, MediaSource};
use nshare::AsNdarray3Mut;
use thiserror::Error;
use tracing::{debug, error, warn};

use std::fs::{metadata, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

//define meadia error
#[derive(Error, Debug)]
pub enum MediaError {
    #[error("Failed to open file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to decode: {0}")]
    ImageDecodeError(#[from] jpeg_decoder::Error),

    #[error("Failed to decode: {0}")]
    VideoDecodeError(String),

    #[error("Ffmpeg error when decoding {1}: {0}")]
    FfmpegError(String, String),
}

pub struct Frame {
    pub file: FileItem,
    pub data: Array3<f32>,
    pub width: usize,
    pub height: usize,
    pub padding: (usize, usize),
    pub ratio: f32,
    pub frame_index: usize,
    pub total_frames: usize,
    pub shoot_time: Option<DateTime<Local>>,
}

pub struct ErrFile {
    pub file: FileItem,
    pub error: anyhow::Error,
}

pub enum ArrayItem {
    Frame(Frame),
    ErrFile(ErrFile),
}

pub fn media_worker(
    file: FileItem,
    imgsz: usize,
    iframe: bool,
    max_frames: Option<usize>,
    array_q_s: Sender<ArrayItem>,
) {
    let mut parser = MediaParser::new();
    let mut resizer = Resizer::new();
    if let Some(extension) = file.file_path.extension() {
        let array_q_s = array_q_s.clone();
        match extension.to_str().unwrap().to_lowercase().as_str() {
            "jpg" | "jpeg" | "png" => {
                process_image(&file, imgsz, &mut parser, &mut resizer, array_q_s).unwrap();
            }
            "mp4" | "avi" | "mkv" | "mov" => {
                process_video(&file, imgsz, iframe, max_frames, array_q_s).unwrap();
            }
            _ => (),
        }
        if &file.file_path != &file.tmp_path {
            remove_file_with_retries(&file.tmp_path, 3, Duration::from_secs(1))
                .expect("Failed to remove file");
        }
    }
}

fn remove_file_with_retries(file_path: &PathBuf, max_retries: u32, delay: Duration) -> Result<()> {
    let mut attempts = 0;

    while attempts < max_retries {
        match std::fs::remove_file(file_path) {
            Ok(_) => {
                debug!("File removed successfully.");
                return Ok(());
            }
            Err(e) => {
                error!(
                    "Failed to remove file: {}. Attempt {} of {}",
                    e,
                    attempts + 1,
                    max_retries
                );
                attempts += 1;

                if attempts < max_retries {
                    thread::sleep(delay);
                }
            }
        }
    }

    Ok(())
}

fn decode_image(file: &FileItem) -> Result<DynamicImage> {
    let img = match ImageReader::open(file.tmp_path.as_path())
        .map_err(MediaError::IoError)?
        .decode()
    {
        Ok(img) => img,
        Err(_e) => {
            warn!(
                "Failed to decode image with ImageReader. Trying jpeg_decoder. {:?}",
                _e
            );
            let img_reader = File::open(file.tmp_path.as_path()).map_err(MediaError::IoError)?;
            let mut decoder = Decoder::new(BufReader::new(img_reader));
            let pixels = decoder.decode().map_err(MediaError::ImageDecodeError)?;
            let img = DynamicImage::ImageRgb8(
                image::ImageBuffer::from_raw(
                    decoder.info().unwrap().width as u32,
                    decoder.info().unwrap().height as u32,
                    pixels,
                )
                .unwrap(),
            );
            img
        }
    };
    Ok(img)
}

pub fn process_image(
    file: &FileItem,
    imgsz: usize,
    parser: &mut MediaParser,
    resizer: &mut Resizer,
    array_q_s: Sender<ArrayItem>,
) -> Result<()> {
    let frame_data = match decode_image(file) {
        Ok(img) => {
            let (img_array, pad_w, pad_h, ratio) = resize_with_pad(&img, imgsz as u32, resizer)?;
            let shoot_time: Option<DateTime<Local>> =
                match get_image_date(parser, file.tmp_path.as_path()) {
                    Ok(shoot_time) => Some(shoot_time),
                    Err(_e) => None,
                };
            let frame_data = Frame {
                data: img_array,
                file: file.clone(),
                width: img.width() as usize,
                height: img.height() as usize,
                padding: (pad_w, pad_h),
                ratio,
                frame_index: 0,
                total_frames: 1,
                shoot_time,
            };

            ArrayItem::Frame(frame_data)
        }
        Err(error) => ArrayItem::ErrFile(ErrFile {
            file: file.clone(),
            error,
        }),
    };
    array_q_s.send(frame_data).expect("Send image frame failed");

    Ok(())
}

fn resize_with_pad(
    img: &DynamicImage,
    imgsz: u32,
    resizer: &mut Resizer,
) -> Result<(Array3<f32>, usize, usize, f32)> {
    // Get the dimensions of the original image
    let (width, height) = img.dimensions();
    let mut resized_width = imgsz;
    let mut resized_height = imgsz;
    let ratio: f32;

    if width > height {
        ratio = width as f32 / imgsz as f32;
        resized_height = (height as f32 / ratio) as u32;
        resized_height = resized_height % 2 + resized_height;
    } else {
        ratio = height as f32 / imgsz as f32;
        resized_width = (width as f32 / ratio) as u32;
        resized_width = resized_width % 2 + resized_width;
    }

    let mut resized_img = DynamicImage::new(resized_width, resized_height, img.color());

    let resize_option = ResizeOptions::new().resize_alg(ResizeAlg::Nearest);

    resizer
        .resize(img, &mut resized_img, &resize_option)
        .unwrap();

    let mut resized_img = resized_img.to_rgb8();

    let image_array = resized_img.as_ndarray3_mut().mapv(|x| x as f32 / 255.0);

    let pad_width = (imgsz - resized_width) / 2;
    let pad_height = (imgsz - resized_height) / 2;

    let mut padded_array = Array3::<f32>::from_elem(Dim([3, imgsz as usize, imgsz as usize]), 0.44);

    padded_array
        .slice_mut(s![
            ..,
            pad_height as i32..(imgsz - pad_height) as i32,
            pad_width as i32..(imgsz - pad_width) as i32
        ])
        .assign(&image_array);

    Ok((padded_array, pad_width as usize, pad_height as usize, ratio))
}

pub fn process_video(
    file: &FileItem,
    imgsz: usize,
    iframe: bool,
    max_frames: Option<usize>,
    array_q_s: Sender<ArrayItem>,
) -> Result<()> {
    let video_path = file.tmp_path.to_string_lossy();
    let input = create_ffmpeg_iter(&video_path, imgsz, iframe)?;

    handle_ffmpeg_output(input, array_q_s, imgsz, file, max_frames)?;

    Ok(())
}

fn create_ffmpeg_iter(video_path: &str, imgsz: usize, iframe: bool) -> Result<FfmpegIterator> {
    let mut ffmpeg_command = FfmpegCommand::new();
    if iframe {
        ffmpeg_command.args(["-skip_frame", "nokey"]);
    }
    let iter = ffmpeg_command
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
        .spawn()?
        .iter()?;
    Ok(iter)
}

fn handle_ffmpeg_output(
    input: FfmpegIterator,
    s: Sender<ArrayItem>,
    imgsz: usize,
    file: &FileItem,
    max_frames: Option<usize>,
) -> Result<()> {
    let file_path = file.file_path.to_string_lossy().into_owned();

    let mut frames = Vec::new();
    let mut ffmpeg_error = Vec::new();
    for event in input {
        match event {
            FfmpegEvent::Error(e) | FfmpegEvent::Log(LogLevel::Error, e) => {
                ffmpeg_error.push(e);
            }
            FfmpegEvent::OutputFrame(frame) => {
                frames.push(frame);
            }
            _ => (),
        }
    }

    for e in ffmpeg_error {
        let error = MediaError::FfmpegError(e, file_path.clone());
        warn!("{:?}", error);
    }

    if frames.is_empty() {
        let error = MediaError::VideoDecodeError(file_path).into();
        error!("{:?}", error);
        let frame_data = ArrayItem::ErrFile(ErrFile {
            file: file.clone(),
            error,
        });
        s.send(frame_data).expect("Send video frame failed");
    } else {
        let sampled_frames = sample_evenly(&frames, max_frames.unwrap_or(frames.len()));

        let shoot_time: Option<DateTime<Local>> = match get_video_date(&file.tmp_path.as_path()) {
            Ok(shoot_time) => Some(shoot_time),
            Err(_e) => None,
        };

        //calculate ratio and padding
        let width = frames[0].width as usize;
        let height = frames[0].height as usize;
        let pad = (width as i32 - height as i32).abs() / 2;
        let padding = if width > height {
            (0, pad as usize)
        } else {
            (pad as usize, 0)
        };
        let ratio = width.max(height) as f32 / imgsz as f32;

        let frames_length = sampled_frames.len();

        for f in sampled_frames.into_iter() {
            let ndarray_frame = Array3::from_shape_vec((imgsz, imgsz, 3), f.data).unwrap();
            let mut ndarray_frame = ndarray_frame.map(|&x| x as f32 / 255.0);
            ndarray_frame = ndarray_frame.permuted_axes([2, 0, 1]);
            let frame_data = ArrayItem::Frame(Frame {
                data: ndarray_frame,
                file: file.clone(),
                width,
                height,
                padding,
                ratio,
                frame_index: f.frame_num as usize,
                total_frames: frames_length,
                shoot_time,
            });
            s.send(frame_data).expect("Send video frame failed");
        }
    }

    Ok(())
}

fn get_image_date(parser: &mut MediaParser, image: &Path) -> Result<DateTime<Local>> {
    let ms = MediaSource::file_path(image)?;

    let iter: ExifIter = parser.parse(ms)?;
    let exif: Exif = iter.into();
    let shoot_time = exif
        .get(ExifTag::DateTimeOriginal)
        .or_else(|| exif.get(ExifTag::ModifyDate))
        .context("Neither DateTimeOriginal nor ModifyDate found")?;
    let shoot_time = shoot_time.as_time().unwrap().with_timezone(&Local);

    Ok(shoot_time)
}

fn get_video_date(video: &Path) -> Result<DateTime<Local>> {
    let metadata = metadata(video)?;
    #[cfg(target_os = "windows")]
    {
        let m_time = metadata.modified()?;
        let shoot_time: DateTime<Local> = m_time.clone().into();

        Ok(shoot_time)
    }

    #[cfg(target_os = "linux")]
    #[allow(deprecated)]
    {
        use chrono::NaiveDateTime;
        use std::os::linux::fs::MetadataExt;
        let m_time: i64 = metadata.st_mtime();
        let c_time: i64 = metadata.st_ctime();
        let shoot_time = m_time.min(c_time);
        let offset = Local::now().offset().to_owned();
        let shoot_time = NaiveDateTime::from_timestamp(shoot_time, 0);
        let shoot_time = DateTime::<Local>::from_naive_utc_and_offset(shoot_time, offset);

        Ok(shoot_time)
    }

    #[cfg(target_os = "macos")]
    {
        use chrono::NaiveDateTime;
        use std::os::unix::fs::MetadataExt;
        let m_time: i64 = metadata.mtime();
        let c_time: i64 = metadata.ctime();
        let shoot_time = m_time.min(c_time);
        let offset = Local::now().offset().to_owned();
        let shoot_time = NaiveDateTime::from_timestamp(shoot_time, 0);
        let shoot_time = DateTime::<Local>::from_naive_utc_and_offset(shoot_time, offset);

        Ok(shoot_time)
    }
}
