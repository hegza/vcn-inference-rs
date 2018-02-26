## Log-level
All programs that are included read the log level from an environment variable
with the key `RUST_LOG`. Here are a couple of examples on how to turn on
logging on linux:
- all: `export RUST_LOG=warn,rusty_cnn=trace`
- high-level: `export RUST_LOG=warn,rusty_cnn=info`

