use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use jpegxr::Decoder;

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: cargo run -p jpegxr@0.1.0 --example inspect -- <image.jxr>");
        return ExitCode::FAILURE;
    };

    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };

    match Decoder::new(&bytes) {
        Ok(decoder) => {
            println!("{:#?}", decoder.info());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
