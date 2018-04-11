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
use std::slice::from_raw_parts_mut;
use byteorder::{LittleEndian, ReadBytesExt};
use layers::Layer;
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
    /// Reads a file into a Vec of Selfs.
    fn read_bin_from_file(filename: &str) -> Vec<Self>;
}

pub trait WriteLinesIntoFile: Sized {
    /// Writes a slice of Self into a file with newline for each Self.
    fn write_lines_into_file(filename: &str, buf: &[Self]);
}

pub trait ReadLinesFromFile: Sized {
    /// Read a vector of Selfs from a file with newline for each Self.
    fn read_lines_from_file(filename: &str) -> Vec<Self>;
}

pub trait ReadCsv: Sized {
    fn read_csv(filename: &str) -> Vec<Self>;
}

pub trait WriteCsv: Sized {
    /// Writes a slice of Selfs into a csv file
    fn write_csv(filename: &str, buf: &[Self]);
}

impl ReadBinFromFile for f32 {
    fn read_bin_from_file(filename: &str) -> Vec<f32> {
        let metadata =
            std::fs::metadata(&filename).expect(&format!("file not found '{}'", filename));

        let f = File::open(filename).expect(&format!("file not found '{}'", filename));
        // f32 = 4 bytes
        let len_f32s = metadata.len() as usize * 4;
        let mut reader = BufReader::with_capacity(len_f32s, f);

        let mut floats: Vec<f32> = Vec::with_capacity(len_f32s + 1);
        // Iterate the file into f32s
        while let Ok(f) = reader.read_f32::<LittleEndian>() {
            floats.push(f);
        }
        floats
    }
}

impl ReadBinFromFile for f64 {
    fn read_bin_from_file(filename: &str) -> Vec<f64> {
        let metadata =
            std::fs::metadata(&filename).expect(&format!("file not found '{}'", filename));

        let f = File::open(filename).expect(&format!("file not found '{}'", filename));
        // f64 = 8 bytes
        let len_f64s = metadata.len() as usize * 8;
        let mut reader = BufReader::with_capacity(len_f64s, f);

        // Iterate the file into f64s
        let mut floats: Vec<f64> = Vec::with_capacity(len_f64s + 1);
        while let Ok(f) = reader.read_f64::<LittleEndian>() {
            floats.push(f);
        }
        floats
    }
}

impl WriteLinesIntoFile for f32 {
    fn write_lines_into_file(filename: &str, f32s: &[f32]) {
        let path: &Path = Path::new(filename);
        let parent: &Path = path.parent().unwrap();
        create_dir_all(parent).unwrap();
        let mut file = File::create(filename).expect("unable to create file");

        for f in f32s {
            write!(file, "{}\n", f).expect("unable to write f32 to file");
        }
    }
}

impl WriteLinesIntoFile for f64 {
    fn write_lines_into_file(filename: &str, f64s: &[f64]) {
        let path: &Path = Path::new(filename);
        let parent: &Path = path.parent().unwrap();
        create_dir_all(parent).unwrap();
        let mut file = File::create(filename).expect("unable to create file");

        for f in f64s {
            write!(file, "{}\n", f).expect("unable to write f64 to file");
        }
    }
}

impl<T> ReadCsv for T
where
    T: Num + FromStr,
    <T as FromStr>::Err: Debug,
{
    fn read_csv(filename: &str) -> Vec<T> {
        let file = File::open(filename).expect(&format!("unable to read file '{}'", filename));
        let chars = BufReader::new(file)
            .lines()
            .filter_map(|res| res.ok())
            .fold(String::new(), |total, line| [total, line].join(""));
        let entries = chars.split(',');
        // Parse into T
        entries
            .map(|e| e.trim().parse::<T>())
            .filter_map(|res| res.ok())
            .collect::<Vec<T>>()
    }
}

impl<T> WriteCsv for T
where
    T: Num + FromStr + std::fmt::Display,
    <T as FromStr>::Err: Debug,
{
    fn write_csv(filename: &str, buf: &[T]) {
        let path: &Path = Path::new(filename);
        let parent: &Path = path.parent().unwrap();
        create_dir_all(parent).unwrap();
        let mut file = File::create(filename).expect("unable to create file");

        for val in buf {
            write!(file, "{},", val).expect("unable to write into file");
        }
    }
}

impl<T> ReadLinesFromFile for T
where
    T: Num + FromStr,
    <T as FromStr>::Err: Debug,
{
    fn read_lines_from_file(filename: &str) -> Vec<T> {
        let file = File::open(filename).expect(&format!("unable to read file '{}'", filename));
        let lines = BufReader::new(file).lines();
        lines
            .into_iter()
            .filter_map(|res| res.ok())
            .map(|line| line.trim().parse::<T>().unwrap())
            .collect::<Vec<T>>()
    }
}

pub trait IndexMatrix<T> {
    fn elem(&self, length: usize, row: usize, column: usize) -> &T;
    fn elem_mut(&mut self, length: usize, row: usize, column: usize) -> &mut T;
}

/// Verifies that each network layer inputs data of valid dimensions to the next layer.
pub fn verify_network_dimensions(layers: &[&Layer]) {
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

/// Splits the input slice into three slices
fn split_in_three_mut<T>(
    slice: &mut [T],
    first_mark: usize,
    second_mark: usize,
) -> (&mut [T], &mut [T], &mut [T]) {
    let len = slice.len();
    let ptr = slice.as_mut_ptr();
    debug_assert!(first_mark <= len);
    debug_assert!(second_mark <= len);
    debug_assert!(first_mark <= second_mark);
    unsafe {
        (
            from_raw_parts_mut(ptr, first_mark),
            from_raw_parts_mut(ptr.offset(first_mark as isize), second_mark - first_mark),
            from_raw_parts_mut(ptr.offset(second_mark as isize), len - second_mark),
        )
    }
}

/// `data` is a vector of Ts that's organized into channels.
pub fn with_edge_padding_by_channel<T>(data: &[T], shape: &ImageGeometry, padding: usize) -> Vec<T>
where
    T: Zero + ReadBinFromFile + Copy,
{
    debug_assert_eq!(data.len(), shape.num_elems());

    let padded_shape = shape.with_padding(padding);
    let padding = padding / 2;

    let mut v: Vec<T> = unsafe { vec![std::mem::uninitialized(); padded_shape.num_elems()] };
    {
        // Divide the image area into single-channel chunks (eg. R, G, B)
        let channels = v.chunks_mut(padded_shape.num_elems_per_channel());
        // Iterate each channel (eg. R, G, B)
        for (c_idx, channel) in channels.enumerate() {
            let mut rows: Vec<&mut [T]> = channel.chunks_mut(padded_shape.side()).collect();
            let (first_rows, n_rows, last_rows) =
                split_in_three_mut(&mut rows, padding, padded_shape.side() - padding);

            // Set the first row elements as 0's
            first_rows
                .iter_mut()
                .for_each(|row| row.iter_mut().for_each(|elem| *elem = Zero::zero()));
            for (row_idx, row) in n_rows.iter_mut().enumerate() {
                let (mut pad_left, mut im_middle, mut pad_right) =
                    split_in_three_mut(row, padding, padded_shape.side() - padding);
                // Pad left side of image with 0's
                pad_left.iter_mut().for_each(|x| *x = Zero::zero());
                // Fill image center with contents
                for (col_idx, elem) in im_middle.iter_mut().enumerate() {
                    *elem = data
                        [c_idx * shape.num_elems_per_channel() + row_idx * shape.side() + col_idx];
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
    debug_assert_eq!(v.len(), padded_shape.num_elems());
    v
}

/// Reads a file into a Vec of f32s and adds the given amount of padding.
pub fn read_image_with_padding_from_bin_in_channels<T>(
    filename: &str,
    padded_image_shape: ImageGeometry,
) -> Vec<T>
where
    T: Zero + ReadBinFromFile + Copy,
{
    let image = T::read_bin_from_file(filename);
    let image_shape = padded_image_shape.unpadded();
    let padding = padded_image_shape.padding();

    with_edge_padding_by_channel(&image, &image_shape, padding)
}

pub fn duration_between(start: Instant, end: Instant) -> f64 {
    let duration = end.duration_since(start);
    duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) * 0.000_000_001f64
}
