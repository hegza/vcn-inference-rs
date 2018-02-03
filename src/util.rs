/*
 * Trivial to understand utility functions that need not clutter other namespaces.
 */
use std::fs::{create_dir_all, File};
use std::path::Path;
use std::io::prelude::*;
use std::io::BufReader;
use byteorder::{LittleEndian, ReadBytesExt};
use layers::Layer;

/// Reads a file into a string.
pub fn read_file(filename: &str) -> String {
    let mut file = File::open(filename).expect("file not found");

    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("something went wrong reading the file");
    contents
}

/// Reads a file into a Vec of f32s.
pub fn read_file_as_f32s(filename: &str) -> Vec<f32> {
    let f = File::open(filename).expect("file not found");
    let mut reader = BufReader::new(f);

    // Iterate the file into f32s
    let mut floats: Vec<f32> = Vec::new();
    while let Ok(f) = reader.read_f32::<LittleEndian>() {
        floats.push(f);
    }
    floats
}

/// Reads a file into a Vec of f32s and verifies that the byte-count of the
/// input file matches with the expected amount of f32s.
pub fn read_file_as_f32s_checked(filename: &str, expected_len: usize) -> Result<Vec<f32>, String> {
    let v = read_file_as_f32s(filename);

    let len = v.len();
    if len != expected_len {
        return Err(format!(
            "expected {} f32s to be read from '{}', but {} were read",
            expected_len, filename, len
        ));
    }
    Ok(v)
}

/// Writes a slice of f32's into a file with newline for each f32.
pub fn write_file_f32s(filename: &str, f32s: &[f32]) {
    let path: &Path = Path::new(filename);
    let parent: &Path = path.parent().unwrap();
    create_dir_all(parent).unwrap();
    let mut file = File::create(filename).expect("unable to create file");

    for f in f32s {
        // TODO: the precision is required only for backwards compatibility with the original version (while debugging)
        write!(file, "{:.6}\n", f).expect("unable to write f32 to file");
    }
}

/// Verifies that each network layer inputs data of valid dimensions to the next layer.
pub fn verify_network_dimensions(layers: &[&Layer<f32>]) {
    for w in layers.windows(2) {
        debug_assert_eq!(w[0].num_out(), w[1].num_in());
    }
}

pub trait IndexMatrix<T> {
    fn elem(&self, length: usize, row: usize, column: usize) -> &T;
    fn elem_mut(&mut self, length: usize, row: usize, column: usize) -> &mut T;
}

impl<T> IndexMatrix<T> for [T] {
    fn elem(&self, length: usize, row: usize, column: usize) -> &T {
        &self[row * length + column]
    }
    fn elem_mut(&mut self, length: usize, row: usize, column: usize) -> &mut T {
        &mut self[row * length + column]
    }
}
