use image;
use image::{GenericImage, Pixel};
use std::io;
use std::fs::*;
use std::path::*;
use rusty_cnn::*;
use rusty_cnn::geometry::*;
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

pub fn load_test_data<P, F, T>(
    dir: P,
    class_dirs: &[String],
    load_fun: F,
    input_shape: &ImageGeometry,
) -> Vec<(Vec<T>, Class)>
where
    P: AsRef<Path>,
    F: Fn(&String) -> Vec<T>,
    T: Coeff,
{
    let conv1_filter_shape = PaddedSquare::from_side(CLASSIC_HYPER_PARAMS.conv_1_filter_side);
    let padded_image_shape = input_shape.with_filter_padding(&conv1_filter_shape);
    let padding = padded_image_shape.padding();

    class_dirs
        .iter()
        .flat_map(|class_dir| {
            // Make JPEG-files into vectors of floats in network input-format
            list_files(dir.as_ref().join(class_dir))
                .unwrap()
                .iter()
                .map(|ref file_name| {
                    (
                        // TODO: can be shortened
                        // Load input as a vector of float in the network format
                        with_edge_padding_by_channel(
                            load_fun(&dir.as_ref()
                                .join(class_dir)
                                .join(file_name)
                                .to_str()
                                .unwrap()
                                .to_owned()),
                            &input_shape,
                            padding,
                        ) as Vec<T>,
                        class_dir.parse::<Class>().unwrap(),
                    )
                })
                .collect::<Vec<(Vec<T>, Class)>>()
        })
        .collect::<Vec<(Vec<T>, Class)>>()
}
