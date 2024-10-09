use std::sync::{Arc, Mutex};

use crate::ExportFormat;

use crate::utils::{Bbox, FileItem};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ExportFrame {
    pub file: FileItem,
    pub frame_index: usize,
    pub is_iframe: bool,
    pub bboxes: Vec<Bbox>,
    pub label: String,
    pub error: Option<String>,
}

pub fn export_worker(
    // format: &ExportFormat,
    export_q_r: crossbeam_channel::Receiver<ExportFrame>,
    export_data: &Arc<Mutex<Vec<ExportFrame>>>,
) {
    loop {
        match export_q_r.recv() {
            Ok(export_frame) => {
                export_data.lock().unwrap().push(export_frame);
                // println!("{:?}", export_frame);
            }
            // match format {
            //     ExportFormat::Json => {
            //         let json = serde_json::to_string(&export_frame).unwrap();
            //         println!("{}", json);
            //         export_data.lock().unwrap().push(json);
            //     }
            //     ExportFormat::Csv => {
            //         let csv = export_frame
            //             .bboxes
            //             .iter()
            //             .map(|bbox| {
            //                 format!(
            //                     "{},{},{},{},{},{},{}",
            //                     export_frame.file.file_path.to_string_lossy(),
            //                     export_frame.frame_index,
            //                     bbox.x1,
            //                     bbox.y1,
            //                     bbox.x2,
            //                     bbox.y2,
            //                     bbox.class
            //                 )
            //             })
            //             .collect::<Vec<String>>()
            //             .join("\n");
            //         export_data.lock().unwrap().push(csv.clone());
            //         println!("{}", csv);
            //     }
            // },
            Err(_) => break,
        }
    }
}
