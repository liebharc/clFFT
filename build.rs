#[cfg(feature="build_all")]
extern crate cmake;

#[cfg(feature="build_all")]
use cmake::Config;

fn main() {
    // Call cmake to build clFFT
    #[cfg(feature="build_all")]
    {
        let _ = Config::new("src")
                 .cflag("--build")
                 .build();
    }
}