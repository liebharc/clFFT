extern crate clfft;
extern crate ocl;

use clfft::*;
use ocl::{ProQue, Buffer};
use ocl::builders::ProgramBuilder;
use ocl::flags;

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
    
    // Make a plan
    let plan = 
        builder::<f64>()
        .precision(Precision::Precise)
        .dims([source.len() / 2])
        .input_layout(Layout::ComplexInterleaved)
        .output_layout(Layout::ComplexInterleaved)
        .bake_out_of_place_plan(&ocl_pq).unwrap();
        
    // Execute plan
    plan.enqueue(Direction::Forward, &mut in_buffer, &mut res_buffer).unwrap();
    
    // Wait for calculation to finish and read results
    res_buffer.cmd()
        .read(&mut source)
        .enq()
        .expect("Transferring result vector from the GPU back to memory failed");
        
        
    ocl_pq.queue().finish();
    println!("{:?}", source);
}