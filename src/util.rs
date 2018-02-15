/*
 * Trivial to understand utility functions that need not clutter other namespaces.
*/
#![allow(dead_code)]
use std;
use std::fs::{create_dir_all, File};
use std::path::Path;
use std::io::prelude::*;
use std::io::BufReader;
use std::time::Instant;
use std::str::FromStr;
use std::fmt::Debug;
use byteorder::{LittleEndian, ReadBytesExt};
use network::Layer;
use geometry::{ImageGeometry, Square};
use num_traits::{Num, Zero};
use Coeff;

/// Reads a file into a string.
pub fn read_file(filename: &str) -> String {
    let mut file = File::open(filename).expect("file not found");

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("something went wrong reading the file");
    contents
}

pub trait ReadBinFromFile: Sized {
    fn read_bin_from_file(filename: &str) -> Vec<Self>;
}

pub trait WriteIntoFile: Sized {
    fn write_into_file(filename: &str, buf: &[Self]);
}

pub trait ReadFromFile: Sized {
    fn read_from_file(filename: &str) -> Vec<Self>;
}

pub trait IndexMatrix<T> {
    fn elem(&self, length: usize, row: usize, column: usize) -> &T;
    fn elem_mut(&mut self, length: usize, row: usize, column: usize) -> &mut T;
}

impl ReadBinFromFile for f32 {
    /// Reads a file into a Vec of f32s.
    fn read_bin_from_file(filename: &str) -> Vec<f32> {
        let f = File::open(filename).expect("file not found");
        let mut reader = BufReader::new(f);

        // Iterate the file into f32s
        let mut floats: Vec<f32> = Vec::new();
        while let Ok(f) = reader.read_f32::<LittleEndian>() {
            floats.push(f);
        }
        floats
    }
}

impl WriteIntoFile for f32 {
    /// Writes a slice of f32s into a file with newline for each f32.
    fn write_into_file(filename: &str, f32s: &[f32]) {
        let path: &Path = Path::new(filename);
        let parent: &Path = path.parent().unwrap();
        create_dir_all(parent).unwrap();
        let mut file = File::create(filename).expect("unable to create file");

        for f in f32s {
            write!(file, "{}\n", f).expect("unable to write f32 to file");
        }
    }
}

impl<T> ReadFromFile for T
where
    T: Num + FromStr,
    <T as FromStr>::Err: Debug,
{
    /// Read a vector of Ts from a file with newline for each T.
    fn read_from_file(filename: &str) -> Vec<T> {
        let file = File::open(filename).expect("unable to create file");
        let lines = BufReader::new(file).lines();
        lines
            .into_iter()
            .filter_map(|res| res.ok())
            .map(|line| line.trim().parse::<T>().unwrap())
            .collect::<Vec<T>>()
    }
}

/// Verifies that each network layer inputs data of valid dimensions to the next layer.
pub fn verify_network_dimensions<T>(layers: &[&Layer<T>])
where
    T: Coeff,
{
    for w in layers.windows(2) {
        debug_assert_eq!(w[0].num_out(), w[1].num_in());
    }
}

impl<T> IndexMatrix<T> for [T] {
    fn elem(&self, length: usize, row: usize, column: usize) -> &T {
        &self[row * length + column]
    }
    fn elem_mut(&mut self, length: usize, row: usize, column: usize) -> &mut T {
        &mut self[row * length + column]
    }
}

/// Reads a file into a Vec of f32s and adds the given amount of padding.
pub fn read_image_with_padding<T>(filename: &str, padded_image_shape: ImageGeometry) -> Vec<T>
where
    T: Zero + ReadBinFromFile + Copy,
{
    let image = T::read_bin_from_file(filename);
    let image_shape = padded_image_shape.unpadded();

    debug_assert_eq!(image.len(), image_shape.num_elems());
    let padding = (padded_image_shape.side() - image_shape.side()) / 2;

    // TODO: There's some room to optimize here, at least in terms of visual pleasure :)
    let mut v: Vec<T> =
        unsafe { vec![std::mem::uninitialized(); padded_image_shape.num_elems()] };
    {
        let channels = v.chunks_mut(padded_image_shape.num_elems() / padded_image_shape.channels());
        for (c, channel) in channels.enumerate() {
            let mut rows: Vec<&mut [T]> = channel.chunks_mut(padded_image_shape.side()).collect();
            let (first_rows, other_rows) = rows.split_at_mut(padding);
            let (n_rows, last_rows) = other_rows.split_at_mut(image_shape.side());

            // Set the first row elements as 0's
            first_rows
                .iter_mut()
                .for_each(|row| row.iter_mut().for_each(|elem| *elem = Zero::zero()));
            for (row_idx, row) in n_rows.iter_mut().enumerate() {
                let (mut pad_left, mut right) = row.split_at_mut(padding);
                let (mut im_middle, mut pad_right) = right.split_at_mut(image_shape.side());
                // Pad left side of image with 0's
                pad_left.iter_mut().for_each(|x| *x = Zero::zero());
                // Fill image center with contents
                for (col_idx, elem) in im_middle.iter_mut().enumerate() {
                    *elem = image[c * (image_shape.num_elems() / padded_image_shape.channels())
                                      + row_idx * image_shape.side()
                                      + col_idx];
                }
                // Pad right side of image with 0's
                pad_right.iter_mut().for_each(|x| *x = Zero::zero());
            }
            // Set the last rows elements as 0's
            last_rows
                .iter_mut()
                .for_each(|row| row.iter_mut().for_each(|elem| *elem = Zero::zero()));
        }
    }
    debug_assert_eq!(v.len(), padded_image_shape.num_elems());
    v
}

pub fn duration_between(start: Instant, end: Instant) -> f64 {
    let duration = end.duration_since(start);
    duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 0.000000001f64
}
