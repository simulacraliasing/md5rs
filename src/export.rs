use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::utils::{Bbox, FileItem};
use crate::ExportFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFrame {
    pub file: FileItem,
    pub shoot_time: Option<String>,
    pub frame_index: usize,
    pub total_frames: usize,
    pub is_iframe: bool,
    pub bboxes: Option<Vec<Bbox>>,
    pub label: Option<String>,
    pub error: Option<String>,
}

pub fn parse_export_csv<P: AsRef<Path>>(csv: P) -> Result<Vec<ExportFrame>> {
    let file = File::open(csv)?;
    let mut rdr = csv::Reader::from_reader(file);
    let mut export_data = Vec::new();
    for frame in rdr.records() {
        let frame = frame?;
        let file_item = FileItem {
            folder_id: frame[0].parse::<_>()?,
            file_id: frame[1].parse::<_>()?,
            file_path: frame[2].parse()?,
        };
        let bboxes = frame[7].to_string().replace("\"\"", "\"");
        let bboxes = serde_json::from_str(&bboxes)?;
        let frame_item = ExportFrame {
            file: file_item,
            shoot_time: Some(frame[3].to_string()),
            frame_index: frame[4].parse::<_>()?,
            total_frames: frame[5].parse::<_>()?,
            is_iframe: frame[6].parse::<bool>()?,
            bboxes: bboxes,
            label: Some(frame[8].to_string()),
            error: Some(frame[9].to_string()),
        };
        export_data.push(frame_item);
    }
    Ok(export_data)
}

pub fn export_worker(
    checkpoint: usize,
    checkpoint_counter: &Arc<Mutex<usize>>,
    format: &ExportFormat,
    folder_path: &str,
    export_q_r: crossbeam_channel::Receiver<ExportFrame>,
    export_data: &Arc<Mutex<Vec<ExportFrame>>>,
) {
    loop {
        match export_q_r.recv() {
            Ok(export_frame) => {
                let mut checkpoint_counter = checkpoint_counter.lock().unwrap();
                *checkpoint_counter += 1;
                if *checkpoint_counter % checkpoint == 0 {
                    let export_data = export_data.lock().unwrap();
                    match format {
                        ExportFormat::Json => write_json(&export_data, folder_path).unwrap(),
                        ExportFormat::Csv => write_csv(&export_data, folder_path).unwrap(),
                    }
                }
                export_data.lock().unwrap().push(export_frame);
            }
            Err(_) => break,
        }
    }
}

fn write_json(export_data: &Vec<ExportFrame>, folder_path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(export_data).unwrap();
    let json_path = std::path::Path::new(&folder_path).join("result.json");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(json_path)
        .unwrap();
    file.write_all(json.as_bytes()).unwrap();
    Ok(())
}

fn write_csv(export_data: &Vec<ExportFrame>, folder_path: &str) -> Result<()> {
    let csv = export_data
        .iter()
        .map(|export_frame| {
            format!(
                "{},{},{},{},{},{},{},{},{},{}",
                export_frame.file.folder_id,
                export_frame.file.file_id,
                export_frame.file.file_path.to_string_lossy(),
                export_frame
                    .shoot_time
                    .clone()
                    .unwrap_or("null".to_string()),
                export_frame.frame_index,
                export_frame.total_frames,
                export_frame.is_iframe,
                format!(
                    "\"{}\"",
                    serde_json::to_string(&export_frame.bboxes)
                        .unwrap_or("null".to_string())
                        .replace("\"", "\"\"")
                ),
                export_frame.label.clone().unwrap_or("null".to_string()),
                format!(
                    "\"{}\"",
                    export_frame
                        .error
                        .clone()
                        .unwrap_or("null".to_string())
                        .replace("\"", "\"\"")
                )
            )
        })
        .collect::<Vec<String>>()
        .join("\n");
    let csv_path = std::path::Path::new(&folder_path).join("result.csv");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(csv_path)
        .unwrap();
    file.write_all(
        "folder_id,file_id,file_path,shoot_time,frame_index,total_frames,is_iframe,bboxes,label,error\n"
            .as_bytes(),
    )
    .unwrap();
    file.write_all(csv.as_bytes()).unwrap();
    Ok(())
}

pub fn export(
    folder_path: &str,
    // checkpoint: usize,
    export_data: Arc<Mutex<Vec<ExportFrame>>>,
    export_format: &ExportFormat,
) -> Result<()> {
    match export_format {
        ExportFormat::Json => {
            let export_data = Arc::try_unwrap(export_data).unwrap().into_inner().unwrap();
            write_json(&export_data, folder_path)?;
        }
        ExportFormat::Csv => {
            let export_data = Arc::try_unwrap(export_data).unwrap().into_inner().unwrap();
            write_csv(&export_data, folder_path)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_export_csv() {
        // let csv = Path::new("input/result.csv");
        let export_data = parse_export_csv("input/result.csv").unwrap();
        assert_eq!(export_data.len(), 11);
    }
}
