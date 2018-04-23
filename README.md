## Installation
1. Install the vendor specific OpenCL drivers and SDK.
2. Install the Rust toolchain by following the instructions at [www.rustup.rs/]().
3. Ask a good mate to provide you with some suitable input data at rusty-cnn/input. File locations are all hard-coded so don't try to do this at home.
4. Compile and run the accuracy tests with `cargo run`, or unit tests with `cargo test` or benchmarks with `cargo bench`. Have a GPU for the last one or you're gonna have a bad time.

Also, Windows users need to make sure to have OpenCL.lib in the project root directory since I don't think they've yet to figure out a standard place to put their libraries on Windows.

## Log-level
All programs that are included read the log level from an environment variable
with the key `RUST_LOG`. Here are a couple of examples on how to turn on
logging on a given terminal:
- all: `export RUST_LOG=warn,rusty_cnn=trace`
- high-level: `export RUST_LOG=warn,rusty_cnn=info`

## Bonus
Run clippy to find things I might've missed. Requires cargo clippy subcommand.
`cargo +nightly clippy`

