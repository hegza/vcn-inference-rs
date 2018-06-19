#![allow(dead_code)]
use image;
use image::{GenericImage, Pixel};
use std::io;
use std::fs::*;
use std::path::*;
use rusty_cnn::*;
use rusty_cnn::math;
use super::class::Class;

pub fn load_jpeg<P>(file: P) -> Vec<f32>
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
                .map(|file_name| {
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
pub fn convert_csv<S, D, ProcessF>(source: &str, destination: &str, cb: ProcessF)
where
    S: ReadCsv,
    D: WriteCsv,
    ProcessF: Fn(&S) -> D,
{
    let src = S::read_csv(source);
    let converted = convert_vec(&src, cb);
    D::write_csv(destination, &converted);
}

pub fn convert_vec<S, D, ProcessF>(source: &[S], cb: ProcessF) -> Vec<D>
where
    ProcessF: Fn(&S) -> D,
{
    source.iter().map(cb).collect::<Vec<D>>()
}

pub fn quantize_vec<S, D>(source: &[S]) -> Vec<D>
where
    S: QuantizeInto<D> + GenericOps + Copy,
    D: Copy,
{
    let max = *source
        .iter()
        .max_by(|a, b| a.generic_partial_cmp(b).unwrap())
        .unwrap();
    let min = *source
        .iter()
        .max_by(|a, b| b.generic_partial_cmp(a).unwrap())
        .unwrap();
    source
        .iter()
        .map(|f| f.quantize(min.clone(), max.clone()))
        .collect::<Vec<D>>()
}
