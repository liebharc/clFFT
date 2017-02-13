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
            let error_code: i32 = unsafe { mem::transmute(result) };
            return ocl::Error::err(format!("FOREIGN_CLFFT_ERROR {}\nSee http://clmathlibraries.github.io/clFFT/clFFT_8h.html#a74e303bed132064cfa22c1cce96d2bce for further information", error_code));
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

fn translate_to_fft_dim(dims: ocl::SpatialDims) -> ffi::clfftDim {
    match dims.dim_count() {
        1 => ffi::clfftDim::CLFFT_1D,
        2 => ffi::clfftDim::CLFFT_2D,
        3 => ffi::clfftDim::CLFFT_3D,
        n => panic!("Number of dimensions must be 1, 2 or 3, but it is {}", n)
    }
}

pub struct FftPlan {
    handle: ffi::clfftPlanHandle
}

impl FftPlan {
    /// Create a plan object initialized entirely with default values.
    ///
    /// A plan is a repository of state for calculating FFT's.  Allows the runtime to pre-calculate kernels, programs 
    /// and buffers and associate them with buffers of specified dimensions.
    pub fn default<D: Into<ocl::SpatialDims>>(pro_que: &ocl::ProQue, dims: D) 
        -> ocl::Result<FftPlan> {
        let context = unsafe { pro_que.context().core_as_ref().as_ptr() };
        let dims = dims.into();
        let dim = translate_to_fft_dim(dims);
        let lengths = try!(dims.to_lens());
        let mut plan: ffi::clfftPlanHandle = 0;
        clfft_try!( unsafe { ffi::clfftCreateDefaultPlan(&mut plan, context, dim, &lengths as *const usize) } );
        Ok(FftPlan {  handle: plan })
    }
    
    /// Returns the native clFFT plan handle.
    pub fn plan_handle(&self) -> ffi::clfftPlanHandle {
        self.handle
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
    use ocl::{ProQue};
    use ocl::builders::ProgramBuilder;
    
    #[test]
    pub fn fft_test() {
        init_lib();
        let prog_bldr = ProgramBuilder::new();
        let ocl_pq = ProQue::builder()
            .prog_bldr(prog_bldr)
            .build()
            .expect("Building ProQue");
        FftPlan::default(&ocl_pq, [1000]);
        drop_lib();
    }
}