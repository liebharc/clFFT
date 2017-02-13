extern crate ocl;
#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
pub mod ffi;
use std::mem;
pub use ffi::clfftSetupData;

macro_rules! clfft_try {
    ( $result_expr: expr) => {
        let result = $result_expr;
        if result != ffi::clfftStatus::CLFFT_SUCCESS {
            return ocl::Error::err_status(unsafe { mem::transmute(result) }, "FOREIGN_CLFFT_ERROR", "See http://clmathlibraries.github.io/clFFT/clFFT_8h.html#a74e303bed132064cfa22c1cce96d2bce for further information");
        }
    }
}

macro_rules! clfft_panic {
    ( $result_expr: expr) => {
        let result = $result_expr;
        if result != ffi::clfftStatus::CLFFT_SUCCESS {
            panic!("Unexpeced error in foreign library: {:?}", result);
        }
    }
}

/// Initialize the internal FFT resources such as FFT implementation caches kernels, programs, and buffers.
pub fn init_lib() -> ffi::clfftSetupData {
    let mut major: ocl::ffi::cl_uint = 0;
    let mut minor: ocl::ffi::cl_uint = 0;
    let mut patch: ocl::ffi::cl_uint = 0;
    clfft_panic!( unsafe { ffi::clfftGetVersion(&mut major, &mut minor, &mut patch) } );
    let data = ffi::clfftSetupData {
        major: major,
        minor: minor,
        patch: patch,
        debugFlags: 0,
    };
    
    clfft_panic!(unsafe { ffi::clfftSetup(&data) });
    data        
}

/// Release all internal resources acquired during `init_lib`.
pub fn drop_lib() {
    unsafe { ffi::clfftTeardown() };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    pub fn fft_test() {
        init_lib();
        
        drop_lib();
    }
}