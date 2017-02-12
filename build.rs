extern crate cmake;
extern crate bindgen;

use std::env;
use std::path::Path;

use cmake::Config;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    /*let _ = bindgen::builder()
    .no_unstable_rust()
    .header("src/include/clFFT.h")
    .hide_type("cl_.*")
    .generate().unwrap()
    .write_to_file(Path::new(&out_dir).join("ffi.rs"));*/
    // Call cmake to build clFFT
    /*let _ = Config::new("src")
                 .cflag("--build")
                 .build();*/
}