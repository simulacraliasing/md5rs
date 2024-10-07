use std::fs::File;
use std::io::Write;

fn main() -> std::io::Result<()> {
    // Define the width, height, and number of frames
    let width = 640;
    let height = 480;
    let frames = 30; // Number of frames for the video

    // Create RGB data for a simple pattern
    let mut buffer = Vec::new();
    let colors = vec![
        [255, 0, 0], // Red
        [0, 255, 0], // Green
        [0, 0, 255], // Blue
    ];

    for _ in 0..frames {
        let mut height_count = 0;
        for _ in 0..height {
            for _ in 0..width {
                buffer.extend_from_slice(&colors[height_count % 3]); // Blue color
            }
            height_count += 1;
        }
    }

    // Write the buffer to a file
    let mut file = File::create("test_video.rgb")?;
    file.write_all(&buffer)?;

    println!("Raw video data created successfully!");

    Ok(())
}
