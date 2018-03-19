#![allow(dead_code)]
use image;
use image::{GenericImage, Pixel};
use std::io;
use std::fs::*;
use std::path::*;
use rusty_cnn::*;
use super::class::Class;

pub fn load_jpeg<P>(file: P) -> Vec<f32>
where
    P: AsRef<Path>,
{
    // TODO: find out local min + max
    let img = image::open(file).unwrap();
    let num_pixels = (img.width() * img.height()) as usize;
    const UMAX: f32 = 255f32;
    const UMIN: f32 = 0f32;
    let mut red_channel = Vec::with_capacity(num_pixels);
    let mut blue_channel = Vec::with_capacity(num_pixels);
    let mut green_channel = Vec::with_capacity(num_pixels);
    for pixel in img.pixels() {
        let rgb = pixel.2.to_rgb();
        let r = rgb[0] as f32 / UMAX + UMIN;
        let g = rgb[1] as f32 / UMAX + UMIN;
        let b = rgb[2] as f32 / UMAX + UMIN;
        red_channel.push(r);
        blue_channel.push(g);
        green_channel.push(b);
    }
    red_channel
        .into_iter()
        .chain(blue_channel)
        .chain(green_channel)
        .collect::<Vec<f32>>()
}

pub fn list_dirs<P>(dir: P) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let mut dirs = Vec::new();
    for entry in read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            dirs.push(entry.file_name().into_string().unwrap());
        }
    }
    Ok(dirs)
}

/// Lists out the full filepaths to all files in the target directory.
pub fn list_files<P>(dir: P) -> io::Result<Vec<String>>
where
    P: AsRef<Path>,
{
    let mut files = Vec::new();
    for entry in read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            files.push(entry.file_name().into_string().unwrap());
        }
    }
    Ok(files)
}

pub fn load_test_data<P, LoadF, T>(
    dir: P,
    class_dirs: &[String],
    load_fun: LoadF,
) -> Vec<(Vec<T>, Class)>
where
    P: AsRef<Path>,
    LoadF: Fn(&String) -> Vec<T>,
    T: Coeff,
{
    class_dirs
        .iter()
        .flat_map(|class_dir| {
            // Make JPEG-files into vectors of floats in network input-format
            list_files(dir.as_ref().join(class_dir))
                .unwrap()
                .iter()
                .map(|ref file_name| {
                    (
                        load_fun(&dir.as_ref()
                            .join(class_dir)
                            .join(file_name)
                            .to_str()
                            .unwrap()
                            .to_owned()),
                        class_dir.parse::<Class>().unwrap(),
                    )
                })
                .collect::<Vec<(Vec<T>, Class)>>()
        })
        .collect::<Vec<(Vec<T>, Class)>>()
}

pub fn convert_all_bin_to_f(dir: &str) {
    let files = list_files(dir).unwrap();
    // Read and re-write files
    for file in files {
        let full_path_name = format!("{}/{}", dir, file);
        let full_path = Path::new(&full_path_name);
        let ext = full_path.extension().unwrap().to_str().unwrap();

        // Read
        let data = f32::read_bin_from_file(&full_path_name);

        let new_name = full_path.to_str().unwrap().replace(&ext, "f");
        // Write them back as f32 (.f)
        f32::write_lines_into_file(&new_name, &data);
    }
    return;
}
