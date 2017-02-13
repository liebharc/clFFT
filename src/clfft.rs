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

/// Specify the expected precision of each FFT.
pub enum Precision {
    Single,
    Double,
    SingleFast,
    DoubleFast
}

fn translate_precision(precision: Precision) -> ffi::clfftPrecision {
    match precision {
        Precision::Single => ffi::clfftPrecision::CLFFT_SINGLE,
        Precision::Double => ffi::clfftPrecision::CLFFT_DOUBLE,
        Precision::SingleFast => ffi::clfftPrecision::CLFFT_SINGLE_FAST,
        Precision::DoubleFast => ffi::clfftPrecision::CLFFT_DOUBLE_FAST
    }
}

fn translate_precision_back(precision: ffi::clfftPrecision) -> Precision {
    match precision {
        ffi::clfftPrecision::CLFFT_SINGLE => Precision::Single,
        ffi::clfftPrecision::CLFFT_DOUBLE => Precision::Double,
        ffi::clfftPrecision::CLFFT_SINGLE_FAST => Precision::SingleFast,
        ffi::clfftPrecision::CLFFT_DOUBLE_FAST => Precision::DoubleFast,
        ffi::clfftPrecision::ENDPRECISION => panic!("ENDPRECISION should never be returned")
    }
}

/// Specify the expected layouts of the buffers.
pub enum Layout {
    ComplexInterleaved,
    ComplexPlanar,
    HermitianInterleaved,
    HermitianPlanar,
    Real
}

fn translate_layout(layout: Layout) -> ffi::clfftLayout {
    match layout {
        Layout::ComplexInterleaved => ffi::clfftLayout::CLFFT_COMPLEX_INTERLEAVED,
        Layout::ComplexPlanar => ffi::clfftLayout::CLFFT_COMPLEX_PLANAR,
        Layout::HermitianInterleaved => ffi::clfftLayout::CLFFT_HERMITIAN_INTERLEAVED,
        Layout::HermitianPlanar => ffi::clfftLayout::CLFFT_HERMITIAN_PLANAR,
        Layout::Real => ffi::clfftLayout::CLFFT_REAL
    }
}

fn translate_layout_back(layout: ffi::clfftLayout) -> Layout {
    match layout {
        ffi::clfftLayout::CLFFT_COMPLEX_INTERLEAVED => Layout::ComplexInterleaved,
        ffi::clfftLayout::CLFFT_COMPLEX_PLANAR => Layout::ComplexPlanar,
        ffi::clfftLayout::CLFFT_HERMITIAN_INTERLEAVED => Layout::HermitianInterleaved,
        ffi::clfftLayout::CLFFT_HERMITIAN_PLANAR => Layout::HermitianPlanar,
        ffi::clfftLayout::CLFFT_REAL => Layout::Real,
        ffi::clfftLayout::ENDLAYOUT => panic!("ENDLAYOUT should never be returned")
    }
}

/// Specify the expected direction of each FFT, time or the frequency domains
pub enum Direction {
    Forward,
    Backward
}

fn translate_direction(direction: Direction) -> ffi::clfftDirection {
    match direction {
        Direction::Forward => ffi::clfftDirection::CLFFT_FORWARD,
        Direction::Backward => ffi::clfftDirection::CLFFT_BACKWARD
    }
}

fn translate_direction_back(direction: ffi::clfftDirection) -> Direction {
    match direction {
        ffi::clfftDirection::CLFFT_FORWARD => Direction::Forward,
        ffi::clfftDirection::CLFFT_BACKWARD => Direction::Backward,
        ffi::clfftDirection::ENDDIRECTION => panic!("ENDDIRECTION should never be returned")
    }
}

/// pecify wheter the input buffers are overwritten with results
pub enum Location {
    Inplace,
    OutOfPlace
}

fn translate_location(location: Location) -> ffi::clfftResultLocation {
    match location {
        Location::Inplace => ffi::clfftResultLocation::CLFFT_INPLACE,
        Location::OutOfPlace => ffi::clfftResultLocation::CLFFT_OUTOFPLACE
    }
}

fn translate_location_back(location: ffi::clfftResultLocation) -> Location {
    match location {
        ffi::clfftResultLocation::CLFFT_INPLACE => Location::Inplace,
        ffi::clfftResultLocation::CLFFT_OUTOFPLACE => Location::OutOfPlace,
        ffi::clfftResultLocation::ENDPLACE => panic!("ENDPLACE should never be returned")
    }
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
    
    /// Set the floating point precision of the FFT data
    pub fn set_precision(&mut self, precision: Precision) -> ocl::Result<()> {
        let precision = translate_precision(precision);
        clfft_try!(unsafe { ffi::clfftSetPlanPrecision(self.handle, precision) });
        Ok(())
    }
    
    pub fn get_precision(&self) -> ocl::Result<Precision> {
        let mut precision = ffi::clfftPrecision::CLFFT_SINGLE;
        clfft_try!(unsafe { ffi::clfftGetPlanPrecision(self.handle, &mut precision) });
        Ok(translate_precision_back(precision))
    }
    
    /// Set the expected layout of the input and output buffers
    pub fn set_layout(&mut self, input_layout: Layout, output_layout: Layout) -> ocl::Result<()> {
        let input_layout = translate_layout(input_layout);
        let output_layout = translate_layout(output_layout);
        clfft_try!(unsafe { ffi::clfftSetLayout(self.handle, input_layout, output_layout) });
        Ok(())
    }
    
    pub fn get_layout(&self) -> ocl::Result<(Layout, Layout)> {
        let mut input_layout = ffi::clfftLayout::CLFFT_COMPLEX_INTERLEAVED;
        let mut output_layout = ffi::clfftLayout::CLFFT_COMPLEX_INTERLEAVED;
        clfft_try!(unsafe { ffi::clfftGetLayout(self.handle, &mut input_layout, &mut output_layout) });
        Ok((translate_layout_back(input_layout), translate_layout_back(output_layout)))
    }
    
    /// Set whether the input buffers are to be overwritten with results
    pub fn set_result_location(&mut self, location: Location) -> ocl::Result<()> {
        let location = translate_location(location);
        clfft_try!(unsafe { ffi::clfftSetResultLocation(self.handle, location) });
        Ok(())
    }
    
    pub fn get_result_location(&self) -> ocl::Result<Location> {
        let mut location = ffi::clfftResultLocation::CLFFT_INPLACE;
        clfft_try!(unsafe { ffi::clfftGetResultLocation(self.handle, &mut location) });
        Ok(translate_location_back(location))
    }
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