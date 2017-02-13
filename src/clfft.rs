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

pub struct FftPlanBuilder<T: ocl::traits::OclPrm> {
    handle: ffi::clfftPlanHandle,
    data_type: std::marker::PhantomData<T>,
    dims: ocl::SpatialDims
}

/// Specify the expected precision of each FFT.
#[derive(PartialEq)]
pub enum Precision {
    Precise,
    Fast
}

fn translate_precision<T>(precision: Precision) -> ffi::clfftPrecision {
    let is_f64 = std::mem::size_of::<T>() == 8;
    match precision {
        Precision::Precise => 
            if is_f64 { ffi::clfftPrecision::CLFFT_DOUBLE }
            else { ffi::clfftPrecision::CLFFT_SINGLE },
        Precision::Fast => 
            if is_f64 { ffi::clfftPrecision::CLFFT_DOUBLE_FAST }
            else { ffi::clfftPrecision::CLFFT_SINGLE_FAST }
    }
}

fn translate_precision_back(precision: ffi::clfftPrecision) -> Precision {
    match precision {
        ffi::clfftPrecision::CLFFT_SINGLE => Precision::Precise,
        ffi::clfftPrecision::CLFFT_DOUBLE => Precision::Precise,
        ffi::clfftPrecision::CLFFT_SINGLE_FAST => Precision::Fast,
        ffi::clfftPrecision::CLFFT_DOUBLE_FAST => Precision::Fast,
        ffi::clfftPrecision::ENDPRECISION => panic!("ENDPRECISION should never be returned")
    }
}

/// Specify the expected layouts of the buffers.
#[derive(PartialEq)]
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
#[derive(PartialEq)]
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

/// pecify wheter the input buffers are overwritten with results
#[derive(PartialEq)]
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

impl<T: ocl::traits::OclPrm> FftPlanBuilder<T> {
    /// Create a plan object initialized entirely with default values.
    ///
    /// A plan is a repository of state for calculating FFT's.  Allows the runtime to pre-calculate kernels, programs 
    /// and buffers and associate them with buffers of specified dimensions.
    pub fn default<D: Into<ocl::SpatialDims>>(pro_que: &ocl::ProQue, dims: D) 
        -> ocl::Result<FftPlanBuilder<T>> {
        let context = unsafe { pro_que.context().core_as_ref().as_ptr() };
        let dims = dims.into();
        let dim = translate_to_fft_dim(dims);
        let lengths = try!(dims.to_lens());
        let mut plan: ffi::clfftPlanHandle = 0;
        clfft_try!( unsafe { ffi::clfftCreateDefaultPlan(&mut plan, context, dim, &lengths as *const usize) } );
        let mut builder = 
            FftPlanBuilder 
            {  
                handle: plan, 
                data_type: std::marker::PhantomData,
                dims: dims
            };
        try!(builder.set_precision(Precision::Precise));
        Ok(builder)
    }
    
    /// Returns the native clFFT plan handle.
    pub fn plan_handle(&self) -> ffi::clfftPlanHandle {
        self.handle
    }
    
    /// Set the floating point precision of the FFT data
    pub fn set_precision(&mut self, precision: Precision) -> ocl::Result<()> {
        let precision = translate_precision::<T>(precision);
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
    
    pub fn build(mut self, pro_que: &mut ocl::ProQue) -> ocl::Result<FftPlan<T>> {
        let mut queue = unsafe { pro_que.queue().core_as_ref().as_ptr() };
        clfft_try!(unsafe {ffi::clfftBakePlan(self.handle, 1, &mut queue, None, 0 as *mut ::std::os::raw::c_void)});
        let (layout_in, layout_out) = try!(self.get_layout());
        let in_complex = if layout_in == Layout::Real { 1 } else { 2 };
        let out_complex = if layout_out == Layout::Real { 1 } else { 2 };
        let fft_len_in = self.dims.to_len() * in_complex;
        let fft_len_out = self.dims.to_len() * out_complex;
        let location = try!{ self.get_result_location() };
        let plan = 
            FftPlan 
            { 
                fft_input_len: fft_len_in,
                fft_output_len: fft_len_out,
                inplace: location == Location::Inplace,
                handle: self.handle,
                data_type: std::marker::PhantomData
            };
        self.handle = 0;
        Ok(plan)
    }
}

pub struct FftPlan<T: ocl::traits::OclPrm> {
    fft_input_len: usize,
    fft_output_len: usize,
    inplace: bool,
    handle: ffi::clfftPlanHandle,
    data_type: std::marker::PhantomData<T>
}

impl<T: ocl::traits::OclPrm> FftPlan<T> {
    pub fn enqueue(
            &self, 
            direction: Direction, 
            pro_que: &mut ocl::ProQue,
            buffer: &mut ocl::Buffer<T>,
            result: Option<&mut ocl::Buffer<T>>) -> ocl::Result<()> {
        if self.fft_input_len != buffer.dims().to_len() {
            return ocl::Error::err(format!("FFT plan requires that input buffer must have a size of {}", self.fft_input_len));
        }
        
        if self.inplace {
            if result.is_some() {
                return ocl::Error::err("Dubious call, FFT is planned to happen inplace but a result buffer was provided. Either change the FFT to operate out of place or don't pass a result buffer into the `enqueue` method.")
            }
        }
        else {
            if !result.is_some() {
                return ocl::Error::err("FFT is planned to be performed out of place but no result buffer was proided. Either change the FFT to operate inplace or provie a result buffer.");
            }
        }
        
        let mut queue = unsafe { pro_que.queue().core_as_ref().as_ptr() };
        let mut buffer = unsafe { buffer.core_as_ref().as_ptr() };
        let mut result = match result {
            None => 0 as ocl::ffi::cl_mem,
            Some(res) => {
                if self.fft_output_len != res.dims().to_len() {
                    return ocl::Error::err(format!("FFT plan requires that result buffer must have a size of {}", self.fft_input_len));
                }
                unsafe { res.core_as_ref().as_ptr() }
            }
        };
        let direction = translate_direction(direction);
        clfft_try!(
            unsafe { 
                ffi::clfftEnqueueTransform(
                    self.handle,
                    direction,
                    1,
                    &mut queue,
                    0,
                    0 as *const ocl::ffi::cl_event,
                    0 as *mut ocl::ffi::cl_event,
                    &mut buffer,
                    &mut result,
                    0 as ocl::ffi::cl_mem)
            });
        Ok(())
    }
    
    /// Returns the native clFFT plan handle.
    pub fn plan_handle(&self) -> ffi::clfftPlanHandle {
        self.handle
    }
}

impl<T: ocl::traits::OclPrm> std::ops::Drop for FftPlanBuilder<T> {
    fn drop(&mut self) {
        if self.handle != 0 {
            // TODO: let _ = unsafe { ffi::clfftDestroyPlan(&mut self.handle) };
            self.handle = 0;
        }
    }
}

impl<T: ocl::traits::OclPrm> std::ops::Drop for FftPlan<T> {
    fn drop(&mut self) {
        if self.handle != 0 {
            // TODO: let _ = unsafe { ffi::clfftDestroyPlan(&mut self.handle) };
            self.handle = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate rustfft;
    extern crate num;

    use super::*;
    use ocl::{ProQue, Buffer};
    use ocl::builders::ProgramBuilder;
    use ocl::flags;
    use std;
    
    fn to_complex(source: &[f64]) -> Vec<num::Complex<f64>> {
        let mut dest = vec![num::Complex{re: 0.0, im: 0.0}; source.len() / 2];
        for i in 0..source.len()/2 {
            dest[i] = num::Complex::new(source[2 * i], source[2 * i + 1]);
        }
        dest
    }
    
    fn to_real(source: &[num::Complex<f64>]) -> Vec<f64> {
        let mut dest = vec![0.0; source.len() * 2];
        for i in 0..source.len() {
            dest[2 * i] = source[i].re;
            dest[2 * i + 1] = source[i].im;
        }
        dest
    }

    pub fn assert_vector_eq(left: &[f64],
                             right: &[f64],
                             tolerance: f64)
    {
        let mut errors = Vec::new();

        let size_assert_failed = left.len() != right.len();
        if size_assert_failed {
            errors.push(format!("Size difference {} != {}", left.len(), right.len()));
        }

        let len = if left.len() < right.len() {
            left.len()
        } else {
            right.len()
        };
        let mut differences = 0;
        for i in 0..len {
            if (left[i] - right[i]).abs() > tolerance {
                differences += 1;
                if differences <= 10 {
                    errors.push(format!("Difference {} at index {}, left: {} != right: {}",
                                        differences,
                                        i,
                                        left[i],
                                        right[i]));
                }
            }
        }

        if differences > 0 {
            errors.push(format!("Total number of differences: {}/{}={}%",
                                differences,
                                len,
                                differences * 100 / len));
        }

        if differences > 0 || size_assert_failed {
            let all_errors = errors.join("\n");
            let header = "-----------------------".to_owned();
            let full_text = format!("\n{}\n{}\n{}\n", header, all_errors, header);
            panic!(full_text);
        }
    }
    
    #[test]
    pub fn fft_test() {
        init_lib();
        let mut source = vec![0.0; 100];
        for i in 0..source.len()/2 {
            let x = std::f64::consts::PI * 4.0 * (i as f64 / 2.0 / source.len() as f64);
            source[2 * i] = x.sin();
            source[2 * i + 1] = x.sin();
        }
        
        let mut fft = rustfft::FFT::new(source.len() / 2, false);
        let signal = to_complex(&source);
        let mut spectrum = signal.clone();
        fft.process(&signal, &mut spectrum);
        
        let prog_bldr = ProgramBuilder::new();
        let mut ocl_pq = ProQue::builder()
            .prog_bldr(prog_bldr)
            .dims([source.len()])
            .build()
            .expect("Building ProQue");
            
        let mut in_buffer =
            Buffer::new(
                ocl_pq.queue().clone(),
                Some(flags::MEM_READ_WRITE |
                     flags::MEM_COPY_HOST_PTR),
                ocl_pq.dims().clone(),
                Some(&source))
                .expect("Failed to create GPU input buffer");
                
        let mut res_buffer =
            Buffer::<f64>::new(
                ocl_pq.queue().clone(),
                Some(flags::MEM_WRITE_ONLY),
                ocl_pq.dims().clone(),
                None)
                .expect("Failed to create GPU result buffer");
        
        let mut builder = FftPlanBuilder::<f64>::default(&ocl_pq, [source.len() / 2]).unwrap();
        builder.set_precision(Precision::Precise).unwrap();
        builder.set_layout(Layout::ComplexInterleaved, Layout::ComplexInterleaved).unwrap();
        builder.set_result_location(Location::OutOfPlace).unwrap();
        let plan = builder.build(&mut ocl_pq).unwrap();
        plan.enqueue(Direction::Forward, &mut ocl_pq, &mut in_buffer, Some(&mut res_buffer)).unwrap();
        ocl_pq.queue().finish();
        
        res_buffer.cmd()
            .read(&mut source)
            .enq()
            .expect("Transferring result vector from the GPU back to memory failed");
        drop_lib();
        assert_vector_eq(&source, &to_real(&spectrum), 1e-4);
    }
}