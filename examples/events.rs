extern crate clfft;
extern crate ocl;

use clfft::*;
use ocl::{MemFlags, ProQue, Buffer, EventList};


// Our kernel source code:
static KERNEL_SRC: &'static str = r#"
    #define mulc64(a,b) ((double2)((double)mad(-(a).y, (b).y, (a).x * (b).x), (double)mad((a).y, (b).x, (a).x * (b).y)))

    __kernel void multiply_vector(
                __global double2 const* const coeff,
                __global double2* const srcres)
    {
        uint const idx = get_global_id(0);
        srcres[idx] = mulc64(srcres[idx], coeff[idx]);
    }
"#;

fn main() {
    // Prepare some data
    let mut source = vec![0.0; 100];
    for i in 0..source.len()/2 {
        let x = std::f64::consts::PI * 4.0 * (i as f64 / 2.0 / source.len() as f64);
        source[2 * i] = x.sin();
        source[2 * i + 1] = x.cos();
    }
    
    let orig = source.clone();
    
    let mut triang = vec![0.0; source.len()];
    for i in 0..source.len() / 2 {
        let half = source.len() / 4;
        triang[2 * i] = if i < half {
            (half - i) as f64 / half as f64
        } else {
            (i - half) as f64 / half as f64
        };
    }
    
    println!("Frequency response: {:?}", triang);
    
    // Build ocl ProQue
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .dims([source.len()])
        .build()
        .expect("Building ProQue");
        
    // Create buffers
    let mut in_buffer = Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_write().copy_host_ptr())
                .dims(ocl_pq.dims().clone())
                .host_data(&source)
                .build().expect("Failed to create GPU input buffer");
                
    let coef_buffer = Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_only().copy_host_ptr())
                .dims(ocl_pq.dims().clone())
                .host_data(&triang)
                .build().expect("Failed to create GPU input buffer");
    
    // Use events to schedule our kernels.
    // When `fft_finish_event` is signaled 
    // then `start_mul_event` gets triggered.
    // Also when `mul_finish_event` is signaled
    // then `start_ifft_event` gets triggered.
    // That leads to a schedule where first the FFT 
    // is executed, then the multiplication and afterwards
    // the IFFT.
    let mut fft_finish_event = EventList::new();
    let start_mul_event = fft_finish_event.clone();
    let mut mul_finish_event = EventList::new();
    let start_ifft_event = mul_finish_event.clone();
    // Make a plan
    let plan = 
        builder::<f64>()
        .precision(Precision::Precise)
        .dims([source.len() / 2])
        .input_layout(Layout::ComplexInterleaved)
        .output_layout(Layout::ComplexInterleaved)
        .bake_inplace_plan(&ocl_pq).unwrap();
        
    // Execute plan
    let mut plan = plan.enew(&mut fft_finish_event);
    plan
        .enq(Direction::Forward, &mut in_buffer)
        .expect("Enq FFT");
        
    let mul = ocl_pq.create_kernel("multiply_vector").unwrap()
        .arg_buf_named("coef", Some(&coef_buffer))
        .arg_buf_named("srcres", Some(&in_buffer));
    mul.cmd()
        .ewait(&start_mul_event)
        .enew(&mut mul_finish_event)
        .gws([source.len() / 2])
        .enq()
        .expect("Enq Mul");
    
    let mut no_events = EventList::new();
    plan
        .enew(&mut no_events) // Remove the `enew` event by passing an empty list
        .ewait(&start_ifft_event)
        .enq(Direction::Backward, &mut in_buffer)
        .expect("Enq IFFT");
    
    // Wait for calculation to finish and read results
    in_buffer.cmd()
        .read(&mut source)
        .enq()
        .expect("Transferring result vector from the GPU back to memory failed");
        
        
    ocl_pq.queue().finish().unwrap();
    println!("Input data (time): {:?}", orig);
    println!("Output data (time): {:?}", source);
}