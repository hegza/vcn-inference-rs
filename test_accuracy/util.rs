#![allow(dead_code)]
use super::class::Class;
use image;
use image::{GenericImage, Pixel};
use rusty_cnn::*;
use std::fs::*;
use std::io;
use std::path::*;

pub fn load_jpeg_as_f32<P>(file: P) -> Vec<f32>
where
    P: AsRef<Path>,
{
    let img = image::open(file).unwrap();
    let num_pixels = (img.width() * img.height()) as usize;
    const UMAX: f32 = 255f32;
    const UMIN: f32 = 0f32;
    let mut red_channel = Vec::with_capacity(num_pixels);
    let mut blue_channel = Vec::with_capacity(num_pixels);
    let mut green_channel = Vec::with_capacity(num_pixels);
    for pixel in img.pixels() {
        let rgb = pixel.2.to_rgb();
        let r = f32::from(rgb[0]) / UMAX + UMIN;
        let g = f32::from(rgb[1]) / UMAX + UMIN;
        let b = f32::from(rgb[2]) / UMAX + UMIN;
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

pub fn load_jpeg_as_u8_lossless<P>(file: P) -> Vec<u8>
where
    P: AsRef<Path>,
{
    let img = image::open(file).unwrap();
    let num_pixels = (img.width() * img.height()) as usize;
    let mut red_channel = Vec::with_capacity(num_pixels);
    let mut blue_channel = Vec::with_capacity(num_pixels);
    let mut green_channel = Vec::with_capacity(num_pixels);
    for pixel in img.pixels() {
        let rgb = pixel.2.to_rgb();
        let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
        red_channel.push(r);
        blue_channel.push(g);
        green_channel.push(b);
    }
    red_channel
        .into_iter()
        .chain(blue_channel)
        .chain(green_channel)
        .collect::<Vec<u8>>()
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

/// Loads test data from per-class subdirectories `class_dirs` of a parent directory `dir` using
/// `load_fun`.
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
                .map(|file_name| {
                    (
                        load_fun(
                            &dir.as_ref()
                                .join(class_dir)
                                .join(file_name)
                                .to_str()
                                .unwrap()
                                .to_owned(),
                        ),
                        class_dir.parse::<Class>().unwrap(),
                    )
                })
                .collect::<Vec<(Vec<T>, Class)>>()
        })
        .collect::<Vec<(Vec<T>, Class)>>()
}

pub fn convert_all_bin_to_f<T>(dir: &str)
where
    T: ReadBinFromFile + WriteLinesIntoFile,
{
    let files = list_files(dir).unwrap();
    // Read and re-write files
    for file in files {
        let full_path_name = format!("{}/{}", dir, file);
        let full_path = Path::new(&full_path_name);
        let ext = full_path.extension().unwrap().to_str().unwrap();

        // Read
        let data = T::read_bin_from_file(&full_path_name);

        let new_name = full_path.to_str().unwrap().replace(&ext, "f");
        // Write them back as T (.f)
        T::write_lines_into_file(&new_name, &data);
    }
    return;
}

/// Converts data of type `S` in `source` into type `D` into file `destination` using `cb` to process the data.
pub fn quantize_csv<S, D, ProcessF>(source: &str, destination: &str, cb: ProcessF)
where
    S: ReadCsv,
    D: WriteCsv,
    ProcessF: Fn(&[S]) -> Vec<D>,
{
    let src = S::read_csv(source);
    let converted = cb(&src);
    D::write_csv(destination, &converted);
}

/*
use ndarray;
pub fn permute_axes<T, A>(vec: Vec<T>, from_shape: ndarray::Shape, to_shape: A) -> Vec<T>
where
    T: ndarray::IntoDimension<Dim = D>,
{
    // Load sparse 3 weights in NCHW
    let sparse3_nchw = f32::read_csv("input/weights/sparse-classic-95-96/fc3-f32-nchw.csv");

    // Create and manipulate representation from_shape to_shape using ndarray
    let mut w = ndarray::Array::from_shape_vec(from_shape, vec).unwrap();
    w = w.permuted_axes(to_shape);

    w.iter().cloned().collect::<Vec<T>>()
}
*/
