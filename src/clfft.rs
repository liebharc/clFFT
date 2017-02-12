extern crate ocl;
#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
pub mod ffi;
/*
use ocl::ffi::*;
use std::os::raw::c_void;

pub type ClFftStatus = i32; // TODO: Make this an enum later on

pub type Context = c_void;

pub type SpatialDims = c_void;

#[repr(C)]
#[allow(dead_code)] // The fields are defined to get the same struct layout as in C
pub struct ClFftSetup {
    major: cl_uint,
    minor: cl_uint,
    patch: cl_uint,
    debug_lags: cl_ulong
}

pub type ClFftPlanHandle = usize;

pub type ClFftDim = i32; // TODO: Make this an enum later on

extern {
    pub fn clfftInitSetupData(setup: &mut ClFftSetup) -> ClFftStatus;
    pub fn clfftSetup(setup: &ClFftSetup) -> ClFftStatus;
    pub fn clfftCreateDefaultPlan(plan: &mut ClFftPlanHandle, 
                                  context: Context,
                                  dim: ClFftDim,
                                  lengths: SpatialDims) -> ClFftStatus;
    pub fn clfftSetPlanPrecision(plan: ClFftPlanHandle, 
    // clfftSetLayout
    // clfftSetResultLocation
    // clfftBakePlan
    // clfftEnqueueTransform
    // clfftDestroyPlan
    pub fn clfftTeardown() -> ClFftStatus;
}*/