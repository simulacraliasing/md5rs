use std::collections::{BTreeMap, HashMap};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::{DateTime, Duration, FixedOffset, Local};
use ndarray::Array;

use crate::export::ExportFrame;

#[derive(Debug, Clone)]
pub struct FileOrg {
    pub folder_id: usize,
    pub file_id: usize,
    pub seq_id: Option<usize>,
    pub move_flag: bool,
    pub dest: Option<PathBuf>,
    pub file_path: PathBuf,
    pub shoot_time: Option<DateTime<FixedOffset>>,
    pub label: Option<String>,
    pub seq_label: Option<String>,
}

impl FileOrg {
    pub fn new(export_frames: Vec<&ExportFrame>) -> Self {
        Self {
            folder_id: export_frames.get(0).unwrap().file.folder_id,
            file_id: export_frames.get(0).unwrap().file.file_id,
            seq_id: None,
            move_flag: false,
            dest: None,
            file_path: export_frames.get(0).unwrap().file.file_path.clone(),
            shoot_time: DateTime::parse_from_rfc3339(
                export_frames
                    .get(0)
                    .unwrap()
                    .to_owned()
                    .clone()
                    .shoot_time
                    .unwrap_or("".to_string())
                    .as_str(),
            )
            .ok(),
            label: get_file_label(export_frames),
            seq_label: None,
        }
    }
}

fn get_file_label(export_frames: Vec<&ExportFrame>) -> Option<String> {
    let label_map = std::collections::HashMap::from([
        ("Animal", 0),
        ("Person", 1),
        ("Vehicle", 2),
        ("Blank", 3),
    ]);
    let mut final_label = None;
    for export_frame in export_frames {
        match &export_frame.label {
            Some(label) => {
                if final_label.is_none() {
                    final_label = Some(label.to_string());
                } else {
                    let a = label_map
                        .get(final_label.clone().unwrap().as_str())
                        .unwrap();
                    let b = label_map.get(label.as_str()).unwrap();
                    if a < b {
                        final_label = Some(label.to_string());
                    }
                }
            }
            None => {}
        }
    }
    final_label
}

fn merge_frames(export_frames: Vec<&ExportFrame>) -> Result<Vec<>>

pub fn organize_frames(export_frames: Vec<ExportFrame>, guess: bool) -> Result<()> {
    let mut folders: BTreeMap<usize, Vec<ExportFrame>> = BTreeMap::new();
    let mut output = Vec::<FileOrg>::new();
    let mut seq_id = 0;

    // let folders = export_frames.iter().map(|f| f.file.folder_id).collect::<Vec<usize>>();

    for export_frame in export_frames {
        let folder_frames = folders.entry(export_frame.file.folder_id).or_insert(vec![]);
        folder_frames.push(export_frame);
    }

    for (_, folder_frames) in folders {
        if folder_frames.len() == 0 {
            continue;
        }
        let folder_path = folder_frames[0].file.file_path.parent().unwrap();
        let animal_folder = folder_path.join("Animal");
        let person_folder = folder_path.join("Person");
        let vehicle_folder = folder_path.join("Vehicle");
        let blank_folder = folder_path.join("Blank");

        let mut is_right_seq = true;
        let mut files = BTreeMap::new();
        let mut files_map: BTreeMap<usize, Vec<&ExportFrame>> = BTreeMap::new();
        for frame in &folder_frames {
            let file_frames = files_map.entry(frame.file.file_id).or_insert(vec![]);
            file_frames.push(frame);
        }
        for (_, frames) in files_map {
            let file_org = FileOrg::new(frames);
            if file_org.label.is_some() {
                files.insert(file_org.file_id, file_org.clone());
            }
            if file_org.shoot_time.is_none() {
                is_right_seq = false;
            }
        }

        if is_right_seq {
            let mut diffs = Vec::new();
            for (i, (_, f)) in files.iter().enumerate() {
                let diff = i as f32 - f.file_id as f32;
                diffs.push(diff);
            }
            let diffs = Array::from_vec(diffs);
            let diffs_std = diffs.std(0.0);
            is_right_seq = diffs_std < 1.0;
        }

        let is_video_time_end = is_video_time_end_time(files.clone()).unwrap_or(false);

        if is_right_seq && !is_video_time_end {
            //placeholder
            let mut seq = vec![];
            for (_, file) in &files {
                if seq.len() == 0 {
                    seq.push(file.clone());
                } else {
                    let duration = (file.shoot_time.unwrap().timestamp()
                        - seq.last().unwrap().shoot_time.unwrap().timestamp())
                    .abs();
                    if duration < 5 {
                        seq.push(file.clone());
                    } else {
                        seq_id += 1;
                    }
                }
            }
            if seq.len() > 0 {
                let (_, output_files) = move_seq(seq, folder_path, seq_id, &mut files)?;
                output.extend(output_files);
            }
        } else if is_right_seq && is_video_time_end && guess {
            let output_files = guess_model(&mut files, folder_path, &mut seq_id)?;
        } else if is_right_seq && is_video_time_end && !guess {
            let output_files = non_guess_model(&mut files, folder_path, &mut seq_id)?;
        } else if !is_right_seq && guess {
            let output_files = guess_model(&mut files, folder_path, &mut seq_id)?;
        } else if !is_right_seq && !guess {
            let output_files = non_guess_model(&mut files, folder_path, &mut seq_id)?;
        }
    }

    Ok(())
}

fn is_video_time_end_time(files: BTreeMap<usize, FileOrg>) -> Result<bool> {
    let mut seq: Vec<FileOrg> = vec![];
    let mut seq_size = vec![];
    for (_, file) in files {
        if seq.len() == 0 {
            seq.push(file);
        } else {
            let duration = (file.shoot_time.unwrap().timestamp()
                - seq.last().unwrap().shoot_time.unwrap().timestamp())
            .abs();
            if duration < 5 {
                seq.push(file);
            } else {
                seq_size.push(seq.len());
                seq = vec![file];
            }
        }
    }
    let array1 = Array::from_vec(seq_size[1..seq_size.len()].to_vec());
    let array2 = Array::from_vec(seq_size[0..seq_size.len() - 1].to_vec());
    let array1 = array1.mapv(|x| x as f32);
    let array2 = array2.mapv(|x| x as f32);
    let mut cross_diff = array1 - array2;
    cross_diff = cross_diff.mapv(|x| x.abs());
    let cross_diff_mean = cross_diff.mean().unwrap();
    Ok(cross_diff_mean > 0.5)
}

fn move_seq(
    seq: Vec<FileOrg>,
    folder_path: &Path,
    seq_id: usize,
    files: &mut BTreeMap<usize, FileOrg>,
) -> Result<(BTreeMap<usize, FileOrg>, Vec<FileOrg>)> {
    let mut  output = Vec::<FileOrg>::new();
    let mut animal_flag = false;
    let mut person_flag = false;
    let mut vehicle_flag = false;
    for file in seq.clone() {
        let label = file.clone().label.unwrap();
        if label == "Animal" {
            animal_flag = true;
            break;
        } else if label == "Person" {
            person_flag = true;
        } else if label == "Vehicle" {
            vehicle_flag = true;
        }
    }
    let label;
    if animal_flag {
        label = "Animal";
    } else if person_flag {
        label = "Person";
    } else if vehicle_flag {
        label = "Vehicle";
    } else {
        label = "Blank";
    }

    for file in seq.clone() {
        let (_,f) = files.remove_entry(&file.file_id).unwrap();
        let mut output_f = f.clone();
        output_f.seq_id = Some(seq_id);
        output_f.seq_label = Some(label.to_string());
        let dest = folder_path
            .join(label)
            .join(&f.file_path.file_name().unwrap());
        match fs::rename(&f.file_path, &dest) {
            Ok(_) => {
                output_f.move_flag = true;
                output_f.dest = Some(dest);
            }
            Err(_) => {
                output_f.move_flag = false;
            }
        }
        output.push(f.clone());
    }

    Ok((files.clone(), output))
}


fn guess_model(
    files: BTreeMap<usize, FileOrg>,
    folder_path: &Path,
    seq_id: &mut usize,
) -> Result<Vec<FileOrg>> {
    let mut output = Vec::<FileOrg>::new();
    let mut files = files.clone();

    // find the best window size
    let file_paths = files
        .iter()
        .map(|(_, f)| f.file_path.clone())
        .collect::<Vec<PathBuf>>();
    let slice_len = if file_paths.len() > 90 {
        90
    } else {
        file_paths.len()
    };
    let first_90_paths = &file_paths[0..slice_len];
    let first_90_suffix = first_90_paths
        .iter()
        .map(|p| p.extension().unwrap())
        .collect::<Vec<&OsStr>>();
    let mut max_count = 0;
    let mut best_window_size = 0;
    let mut non_guess = false;
    for window_size in 1..6 {
        let fold = first_90_suffix.len() / window_size;
        let mut right_count = 0;
        let mut stack = Vec::new();
        for i in (0..window_size * (fold - 1)).step_by(window_size) {
            let window = &first_90_suffix[i..i + window_size];
            if stack.len() == 0 {
                stack = window.to_vec();
            } else if window == stack {
                right_count += 1;
            } else {
                right_count -= 1;
                stack = window.to_vec();
            }
        }
        if right_count > max_count {
            max_count = right_count;
            best_window_size = window_size;
        }
        if window_size >= 5 && best_window_size == 0 {
            non_guess = true;
            break;
        }
    }
    if non_guess {
        let output_files = non_guess_model(files, folder_path, seq_id)?;
        output.extend(output_files);
        return Ok(output);
    }

    let mut offset = 0;
    while first_90_suffix[offset..offset + best_window_size]
        != first_90_suffix[offset + best_window_size..offset + 2 * best_window_size]
    {
        offset += 1;
    }
    let files1 = files.clone().range_mut(0..offset);
    for (_, file) in files1.into_iter() {
        *seq_id += 1;
        file.seq_id = Some(*seq_id);
        file.seq_label = file.label.clone();
        let dest = folder_path
            .join(&file.label.clone().unwrap())
            .join(&file.file_path.file_name().unwrap());
        match fs::rename(&file.file_path, &dest) {
            Ok(_) => {
                file.move_flag = true;
                file.dest = Some(dest);
            }
            Err(_) => {
                file.move_flag = false;
            }
        }
    }
    let mut seq = vec![];
    let mut suffix_stack = vec![];
    let mut new_suffix_stack = vec![];
    let files2 = files.clone().range_mut(offset..files.len());
    for (i, (_, file)) in files2.into_iter().enumerate() {
        if i == 0 {
            seq.push(file.clone());
        } else {
            if i % best_window_size != 0 {
                seq.push(file.clone());
            } else {
                if suffix_stack.len() == 0 {
                    for f in seq.clone() {
                        if let Some(suffix) = f.file_path.extension() {
                            suffix_stack.push(suffix.to_owned());
                        }
                    }
                } else {
                    new_suffix_stack = seq
                        .clone()
                        .iter()
                        .map(|f| f.file_path.extension().unwrap().to_owned())
                        .collect();
                    if new_suffix_stack != suffix_stack {
                        break;
                    }
                }
                *seq_id += 1;
                let (reduced_files, output_files) = move_seq(seq.clone(), folder_path, *seq_id, files)?;
                files = reduced_files;
                output.extend(output_files);
            }
        }
    }
    Ok(output)
}

fn non_guess_model(
    files: BTreeMap<usize, FileOrg>,
    folder_path: &Path,
    seq_id: &mut usize,
) -> Result<Vec<FileOrg>> {
    Ok(vec![])
}
