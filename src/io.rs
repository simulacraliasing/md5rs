use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Ok, Result};
use crossbeam_channel::Sender;
use uuid::Uuid;

use crate::FileItem;

fn copy_to_buff(file_path: &PathBuf, buff_path: &Path) -> Result<PathBuf> {
    let mut tmp_name = Uuid::new_v4().to_string();
    let ext = file_path.extension().unwrap();
    tmp_name.push_str(".");
    tmp_name.push_str(ext.to_str().unwrap());
    let temp_path = buff_path.join(tmp_name);
    fs::copy(file_path, &temp_path)?;
    Ok(temp_path)
}

pub fn io_worker(buff_path: &Path, file: &FileItem, io_q_s: Sender<FileItem>) -> Result<()> {
    let mut new_file = file.clone();
    new_file.tmp_path = copy_to_buff(&file.file_path, buff_path)?;
    io_q_s.send(new_file)?;
    Ok(())
}
