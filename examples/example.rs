extern crate clfft;
extern crate ocl;

use clfft::*;
use ocl::{MemFlags, ProQue, Buffer};
use ocl::builders::ProgramBuilder;

fn main() {
    // Prepare some data
    let mut source = vec![0.0; 100];
    for i in 0..source.len()/2 {
        let x = std::f64::consts::PI * 4.0 * (i as f64 / 2.0 / source.len() as f64);
        source[2 * i] = x.sin();
        source[2 * i + 1] = x.sin();
    }
    
    // Build ocl ProQue
    let prog_bldr = ProgramBuilder::new();
    let ocl_pq = ProQue::builder()
        .prog_bldr(prog_bldr)
        .dims([source.len()])
        .build()
        .expect("Building ProQue");
        
    // Create buffers
    let mut in_buffer = unsafe {
        Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_write().copy_host_ptr())
                .len(ocl_pq.dims().clone())
                .use_host_slice(&source)
                .build().expect("Failed to create GPU input buffer")
				};
            
    let mut res_buffer =
        Buffer::<f64>::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().write_only())
                .len(ocl_pq.dims().clone())
                .build().expect("Failed to create GPU result buffer");
    
    // Make a plan
    let mut plan = 
        builder::<f64>()
        .precision(Precision::Precise)
        .dims([source.len() / 2])
        .input_layout(Layout::ComplexInterleaved)
        .output_layout(Layout::ComplexInterleaved)
        .bake_out_of_place_plan(&ocl_pq).unwrap();
        
    // Execute plan
    plan.enq(Direction::Forward, &mut in_buffer, &mut res_buffer).unwrap();
    
    // Wait for calculation to finish and read results
    res_buffer.cmd()
        .read(&mut source)
        .enq()
        .expect("Transferring result vector from the GPU back to memory failed");
        
        
    ocl_pq.queue().finish().unwrap();
    println!("{:?}", source);
}