use crate::utils::{sample_evenly, FileItem};

use anyhow::Result;
use chrono::{offset, DateTime, Local, NaiveDateTime};
use crossbeam_channel::Sender;
use ffmpeg_sidecar::child::FfmpegChild;
use ffmpeg_sidecar::command::FfmpegCommand;
use ffmpeg_sidecar::event::{FfmpegEvent, LogLevel, OutputVideoFrame};
use image::{DynamicImage, GenericImageView, ImageReader, RgbaImage};
use jpeg_decoder::Decoder;
use ndarray::Array3;
use nom_exif::{Exif, ExifIter, ExifTag, MediaParser, MediaSource};
use thiserror::Error;

use std::fs::{metadata, File};
use std::io::BufReader;
use std::path::Path;

//define meadia error
#[derive(Error, Debug)]
pub enum MediaError {
    #[error("Failed to open file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to decode: {0}")]
    ImageDecodeError(#[from] jpeg_decoder::Error),

    #[error("Failed to decode: {0}")]
    VideoDecodeError(String),
}

pub struct Frame {
    pub file: FileItem,
    pub data: Array3<f32>,
    pub width: usize,
    pub height: usize,
    pub padding: (usize, usize),
    pub ratio: f32,
    pub iframe_index: usize,
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
    if let Some(extension) = file.file_path.extension() {
        let array_q_s = array_q_s.clone();
        match extension.to_str().unwrap().to_lowercase().as_str() {
            "jpg" | "jpeg" | "png" => process_image(file, imgsz, &mut parser, array_q_s).unwrap(),
            "mp4" | "avi" | "mkv" | "mov" => {
                process_video(file, imgsz, iframe, max_frames, array_q_s).unwrap();
            }
            _ => (),
        }
    }
}

fn decode_image(file: &FileItem) -> Result<DynamicImage> {
    let img = match ImageReader::open(file.file_path.as_path())
        .map_err(MediaError::IoError)?
        .decode()
    {
        Ok(img) => img,
        Err(_e) => {
            let img_reader = File::open(file.file_path.as_path()).map_err(MediaError::IoError)?;
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
    file: FileItem,
    imgsz: usize,
    parser: &mut MediaParser,
    array_q_s: Sender<ArrayItem>,
) -> Result<()> {
    let frame_data = match decode_image(&file) {
        Ok(img) => {
            let (img_array, pad_w, pad_h, ratio) = resize_with_pad(&img, imgsz as u32)?;
            let shoot_time: Option<DateTime<Local>> =
                match get_image_date(parser, &file.file_path.as_path()) {
                    Ok(shoot_time) => Some(shoot_time),
                    Err(_e) => None,
                };
            let frame_data = Frame {
                data: img_array,
                file,
                width: img.width() as usize,
                height: img.height() as usize,
                padding: (pad_w, pad_h),
                ratio,
                iframe_index: 0,
                total_frames: 1,
                shoot_time,
            };

            ArrayItem::Frame(frame_data)
        }
        Err(error) => ArrayItem::ErrFile(ErrFile { file, error }),
    };
    array_q_s.send(frame_data).expect("Send image frame failed");

    Ok(())
}

fn resize_with_pad(img: &DynamicImage, imgsz: u32) -> Result<(Array3<f32>, usize, usize, f32)> {
    // Get the dimensions of the original image
    let (width, height) = img.dimensions();

    let ratio = width.max(height) as f32 / imgsz as f32;

    // Calculate the padding needed to make the image square
    let pad_width = ((height as i32 - width as i32).max(0) / 2) as u32;
    let pad_height = ((width as i32 - height as i32).max(0) / 2) as u32;

    let padded_width = width + 2 * pad_width;
    let padded_height = height + 2 * pad_height;
    // Create a new square image with padding
    let mut padded_img = RgbaImage::new(padded_width, padded_height);
    for x in 0..padded_width {
        for y in 0..padded_height {
            if x >= pad_width
                && x < padded_width - pad_width
                && y >= pad_height
                && y < padded_height - pad_height
            {
                let pixel = img.get_pixel(x - pad_width, y - pad_height);
                padded_img.put_pixel(x, y, pixel);
            } else {
                padded_img.put_pixel(x, y, image::Rgba([0, 0, 0, 255])); // Black padding
            }
        }
    }

    // Resize the padded image to the target size
    let resized_img = DynamicImage::ImageRgba8(padded_img).resize_exact(
        imgsz,
        imgsz,
        image::imageops::FilterType::Lanczos3,
    );

    // Convert the image from RGBA to RGB and return
    let mut rgb_ndarray = Array3::zeros((3, imgsz as usize, imgsz as usize));
    for pixel in resized_img.pixels() {
        let (x, y) = (pixel.0 as _, pixel.1 as _);
        let [r, g, b, _] = pixel.2 .0;
        rgb_ndarray[[0, y, x]] = r as f32 / 255.0;
        rgb_ndarray[[1, y, x]] = g as f32 / 255.0;
        rgb_ndarray[[2, y, x]] = b as f32 / 255.0;
    }

    Ok((rgb_ndarray, pad_width as usize, pad_width as usize, ratio))
}

pub fn process_video(
    file: FileItem,
    imgsz: usize,
    iframe: bool,
    max_frames: Option<usize>,
    array_q_s: Sender<ArrayItem>,
) -> Result<()> {
    let video_path = file.file_path.to_string_lossy();
    let input = create_ffmpeg_command(&video_path, imgsz, iframe)?;

    handle_ffmpeg_output(input, array_q_s, imgsz, &file, max_frames)?;

    Ok(())
}

fn create_ffmpeg_command(video_path: &str, imgsz: usize, iframe: bool) -> Result<FfmpegChild> {
    let mut ffmpeg_command = FfmpegCommand::new();
    if iframe {
        ffmpeg_command.args(["-skip_frame", "nokey"]);
    }
    let command = ffmpeg_command
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

fn decode_video(
    mut input: FfmpegChild,
) -> Result<(Vec<OutputVideoFrame>, Option<usize>, Option<usize>)> {
    let mut width = None;
    let mut height = None;
    let mut frames = vec![];

    for e in input.iter()? {
        match e {
            FfmpegEvent::Log(LogLevel::Error, e) => {
                return Err(MediaError::VideoDecodeError(e).into());
            }
            FfmpegEvent::ParsedInputStream(i) => {
                if i.stream_type.to_lowercase() == "video" {
                    width = Some(i.width as usize);
                    height = Some(i.height as usize);
                }
            }
            FfmpegEvent::OutputFrame(f) => {
                frames.push(f);
            }
            _ => {}
        }
    }

    Ok((frames, width, height))
}

fn handle_ffmpeg_output(
    input: FfmpegChild,
    s: Sender<ArrayItem>,
    imgsz: usize,
    file: &FileItem,
    max_frames: Option<usize>,
) -> Result<()> {
    match decode_video(input) {
        Ok((frames, width, height)) => {
            let (sampled_frames, sampled_indexes) =
                sample_evenly(&frames, max_frames.unwrap_or(frames.len()));

            let shoot_time: Option<DateTime<Local>> =
                match get_video_date(&file.file_path.as_path()) {
                    Ok(shoot_time) => Some(shoot_time),
                    Err(_e) => None,
                };

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

            let frames_length = sampled_frames.len();

            for (f, i) in sampled_frames.into_iter().zip(sampled_indexes.into_iter()) {
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
                    iframe_index: i,
                    total_frames: frames_length,
                    shoot_time,
                });
                s.send(frame_data).expect("Send video frame failed");
            }
        }
        Err(error) => {
            let frame_data = ArrayItem::ErrFile(ErrFile {
                file: file.clone(),
                error,
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
        .unwrap_or(exif.get(ExifTag::ModifyDate).unwrap());
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
        let m_time: i64 = metadata.mtime()?;
        let c_time: i64 = metadata.ctime()?;
        let shoot_time = m_time.min(c_time);
        let shoot_time: DateTime<Local> = Local.timestamp(shoot_time, 0);

        Ok(shoot_time)
    }
}
