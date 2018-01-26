/*
 * Trivial to understand utility functions that need not clutter other namespaces.
 */
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use byteorder::{BigEndian, ReadBytesExt};

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
    while let Ok(f) = reader.read_f32::<BigEndian>() {
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
            "expected {} f32s to be read from \"{}\", but {} were read",
            expected_len, filename, len
        ));
    }
    Ok(v)
}
