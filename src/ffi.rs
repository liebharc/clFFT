use ocl::ffi::*;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_platform_id([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_device_id([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_context([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_command_queue([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_mem([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_program([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_kernel([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_event([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _cl_sampler([u8; 0]);
#[repr(C)]
#[derive(Debug, Copy)]
pub struct _cl_image_format {
    pub image_channel_order: cl_channel_order,
    pub image_channel_data_type: cl_channel_type,
}
#[test]
fn bindgen_test_layout__cl_image_format() {
    assert_eq!(::std::mem::size_of::<_cl_image_format>() , 8usize);
    assert_eq!(::std::mem::align_of::<_cl_image_format>() , 4usize);
}
impl Clone for _cl_image_format {
    fn clone(&self) -> Self { *self }
}
#[repr(C)]
#[derive(Debug, Copy)]
pub struct _cl_buffer_region {
    pub origin: usize,
    pub size: usize,
}
#[test]
fn bindgen_test_layout__cl_buffer_region() {
    assert_eq!(::std::mem::size_of::<_cl_buffer_region>() , 16usize);
    assert_eq!(::std::mem::align_of::<_cl_buffer_region>() , 8usize);
}
impl Clone for _cl_buffer_region {
    fn clone(&self) -> Self { *self }
}
#[repr(i32)]
/*   @brief clfft error codes definition(incorporating OpenCL error definitions)
 *
 *   This enumeration is a superset of the OpenCL error codes.  For example, CL_OUT_OF_HOST_MEMORY,
 *   which is defined in cl.h is aliased as CLFFT_OUT_OF_HOST_MEMORY.  The set of basic OpenCL
 *   error codes is extended to add extra values specific to the clfft package.
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum clfftStatus_ {
    CLFFT_INVALID_GLOBAL_WORK_SIZE = -63,
    CLFFT_INVALID_MIP_LEVEL = -62,
    CLFFT_INVALID_BUFFER_SIZE = -61,
    CLFFT_INVALID_GL_OBJECT = -60,
    CLFFT_INVALID_OPERATION = -59,
    CLFFT_INVALID_EVENT = -58,
    CLFFT_INVALID_EVENT_WAIT_LIST = -57,
    CLFFT_INVALID_GLOBAL_OFFSET = -56,
    CLFFT_INVALID_WORK_ITEM_SIZE = -55,
    CLFFT_INVALID_WORK_GROUP_SIZE = -54,
    CLFFT_INVALID_WORK_DIMENSION = -53,
    CLFFT_INVALID_KERNEL_ARGS = -52,
    CLFFT_INVALID_ARG_SIZE = -51,
    CLFFT_INVALID_ARG_VALUE = -50,
    CLFFT_INVALID_ARG_INDEX = -49,
    CLFFT_INVALID_KERNEL = -48,
    CLFFT_INVALID_KERNEL_DEFINITION = -47,
    CLFFT_INVALID_KERNEL_NAME = -46,
    CLFFT_INVALID_PROGRAM_EXECUTABLE = -45,
    CLFFT_INVALID_PROGRAM = -44,
    CLFFT_INVALID_BUILD_OPTIONS = -43,
    CLFFT_INVALID_BINARY = -42,
    CLFFT_INVALID_SAMPLER = -41,
    CLFFT_INVALID_IMAGE_SIZE = -40,
    CLFFT_INVALID_IMAGE_FORMAT_DESCRIPTOR = -39,
    CLFFT_INVALID_MEM_OBJECT = -38,
    CLFFT_INVALID_HOST_PTR = -37,
    CLFFT_INVALID_COMMAND_QUEUE = -36,
    CLFFT_INVALID_QUEUE_PROPERTIES = -35,
    CLFFT_INVALID_CONTEXT = -34,
    CLFFT_INVALID_DEVICE = -33,
    CLFFT_INVALID_PLATFORM = -32,
    CLFFT_INVALID_DEVICE_TYPE = -31,
    CLFFT_INVALID_VALUE = -30,
    CLFFT_MAP_FAILURE = -12,
    CLFFT_BUILD_PROGRAM_FAILURE = -11,
    CLFFT_IMAGE_FORMAT_NOT_SUPPORTED = -10,
    CLFFT_IMAGE_FORMAT_MISMATCH = -9,
    CLFFT_MEM_COPY_OVERLAP = -8,
    CLFFT_PROFILING_INFO_NOT_AVAILABLE = -7,
    CLFFT_OUT_OF_HOST_MEMORY = -6,
    CLFFT_OUT_OF_RESOURCES = -5,
    CLFFT_MEM_OBJECT_ALLOCATION_FAILURE = -4,
    CLFFT_COMPILER_NOT_AVAILABLE = -3,
    CLFFT_DEVICE_NOT_AVAILABLE = -2,
    CLFFT_DEVICE_NOT_FOUND = -1,
    CLFFT_SUCCESS = 0,
    CLFFT_BUGCHECK = 4096,
    CLFFT_NOTIMPLEMENTED = 4097,
    CLFFT_TRANSPOSED_NOTIMPLEMENTED = 4098,
    CLFFT_FILE_NOT_FOUND = 4099,
    CLFFT_FILE_CREATE_FAILURE = 4100,
    CLFFT_VERSION_MISMATCH = 4101,
    CLFFT_INVALID_PLAN = 4102,
    CLFFT_DEVICE_NO_DOUBLE = 4103,
    CLFFT_DEVICE_MISMATCH = 4104,
    CLFFT_ENDSTATUS = 4105,
}
pub use self::clfftStatus_ as clfftStatus;
#[repr(i32)]
/*   @brief The dimension of the input and output buffers that is fed into all FFT transforms */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum clfftDim_ {
    CLFFT_1D = 1,
    CLFFT_2D = 2,
    CLFFT_3D = 3,
    ENDDIMENSION = 4,
}
pub use self::clfftDim_ as clfftDim;
#[repr(i32)]
/*   @brief Specify the expected layouts of the buffers */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum clfftLayout_ {
    CLFFT_COMPLEX_INTERLEAVED = 1,
    CLFFT_COMPLEX_PLANAR = 2,
    CLFFT_HERMITIAN_INTERLEAVED = 3,
    CLFFT_HERMITIAN_PLANAR = 4,
    CLFFT_REAL = 5,
    ENDLAYOUT = 6,
}
pub use self::clfftLayout_ as clfftLayout;
#[repr(i32)]
/*   @brief Specify the expected precision of each FFT.
 */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum clfftPrecision_ {
    CLFFT_SINGLE = 1,
    CLFFT_DOUBLE = 2,
    CLFFT_SINGLE_FAST = 3,
    CLFFT_DOUBLE_FAST = 4,
    ENDPRECISION = 5,
}
pub use self::clfftPrecision_ as clfftPrecision;
pub const clfftDirection__CLFFT_MINUS: clfftDirection_ =
    clfftDirection_::CLFFT_FORWARD;
pub const clfftDirection__CLFFT_PLUS: clfftDirection_ =
    clfftDirection_::CLFFT_BACKWARD;
#[repr(i32)]
/*   @brief Specify the expected direction of each FFT, time or the frequency domains */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum clfftDirection_ {
    CLFFT_FORWARD = -1,
    CLFFT_BACKWARD = 1,
    ENDDIRECTION = 2,
}
pub use self::clfftDirection_ as clfftDirection;
#[repr(i32)]
/*   @brief Specify wheter the input buffers are overwritten with results */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum clfftResultLocation_ {
    CLFFT_INPLACE = 1,
    CLFFT_OUTOFPLACE = 2,
    ENDPLACE = 3,
}
pub use self::clfftResultLocation_ as clfftResultLocation;
#[repr(i32)]
/*  @brief Determines whether the result is returned in original order. It is valid only for
dimensions greater than 1. */
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum clfftResultTransposed_ {
    CLFFT_NOTRANSPOSE = 1,
    CLFFT_TRANSPOSED = 2,
    ENDTRANSPOSED = 3,
}
pub use self::clfftResultTransposed_ as clfftResultTransposed;
/*  @brief Data structure that can be passed to clfftSetup() to control the behavior of the FFT runtime
 *  @details This structure contains values that can be initialized before instantiation of the FFT runtime
 *  with ::clfftSetup().  To initialize this structure, pass a pointer to a user struct to ::clfftInitSetupData( ),
 *  which clears the structure and sets the version member variables to the current values.
 */
#[repr(C)]
#[derive(Debug, Copy)]
pub struct clfftSetupData_ {
    /* < Major version number of the project; signifies possible major API changes. */
    pub major: cl_uint,
    /* < Minor version number of the project; minor API changes that can break backward compatibility. */
    pub minor: cl_uint,
    /* < Patch version number of the project; always incrementing number, signifies change over time. */
    pub patch: cl_uint,
    /*  	Bitwise flags that control the behavior of library debug logic. */
    pub debugFlags: cl_ulong,
}
#[test]
fn bindgen_test_layout_clfftSetupData_() {
    assert_eq!(::std::mem::size_of::<clfftSetupData_>() , 24usize);
    assert_eq!(::std::mem::align_of::<clfftSetupData_>() , 8usize);
}
impl Clone for clfftSetupData_ {
    fn clone(&self) -> Self { *self }
}
pub type clfftSetupData = clfftSetupData_;
#[repr(i32)]
/*  @brief Type of Callback function.
*/
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum clfftCallbackType_ { PRECALLBACK = 0, POSTCALLBACK = 1, }
pub use self::clfftCallbackType_ as clfftCallbackType;
/*   @brief An abstract handle to the object that represents the state of the FFT(s) */
pub type clfftPlanHandle = usize;
extern "C" {
    /*  @brief Initialize the internal FFT resources.
	 *  @details The internal resources include FFT implementation caches kernels, programs, and buffers.
	 *  @param[in] setupData Data structure that is passed into the setup routine to control FFT generation behavior
	 * 	and debug functionality
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftSetup(setupData: *const clfftSetupData) -> clfftStatus;
}
extern "C" {
    /*  @brief Release all internal resources.
	 *  @details Called when client is done with the FFT library, allowing the library to destroy all resources it has cached
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftTeardown() -> clfftStatus;
}
extern "C" {
    /*  @brief Query the FFT library for version information
	 *  @details Returns the major, minor and patch version numbers associated with the FFT library
	 *  @param[out] major Major functionality change
	 *  @param[out] minor Minor functionality change
	 *  @param[out] patch Bug fixes, documentation changes, no new features introduced
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftGetVersion(major: *mut cl_uint, minor: *mut cl_uint,
                           patch: *mut cl_uint) -> clfftStatus;
}
extern "C" {
    /*  @brief Create a plan object initialized entirely with default values.
	 *  @details A plan is a repository of state for calculating FFT's.  Allows the runtime to pre-calculate kernels, programs
	 * 	and buffers and associate them with buffers of specified dimensions.
	 *  @param[out] plHandle Handle to the newly created plan
	 *  @param[in] context Client is responsible for providing an OpenCL context for the plan
	 *  @param[in] dim Dimensionality of the FFT transform; describes how many elements are in the array
	 *  @param[in] clLengths An array of length of size 'dim';  each array value describes the length of each dimension
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftCreateDefaultPlan(plHandle: *mut clfftPlanHandle,
                                  context: cl_context, dim: clfftDim,
                                  clLengths: *const usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Create a copy of an existing plan.
	 *  @details This API allows a client to create a new plan based upon an existing plan.  This function can be used to
	 *  quickly create plans that are similar, but may differ slightly.
	 *  @param[out] out_plHandle Handle to the newly created plan that is based on in_plHandle
	 *  @param[in] new_context Client is responsible for providing a new context for the new plan
	 *  @param[in] in_plHandle Handle to a previously created plan that is to be copied
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftCopyPlan(out_plHandle: *mut clfftPlanHandle,
                         new_context: cl_context,
                         in_plHandle: clfftPlanHandle) -> clfftStatus;
}
extern "C" {
    /*  @brief Prepare the plan for execution.
	 *  @details After all plan parameters are set, the client has the option of 'baking' the plan, which informs the runtime that
	 *  no more change to the parameters of the plan is expected, and the OpenCL kernels can be compiled.  This optional function
	 *  allows the client application to perform the OpenCL kernel compilation when the application is initialized instead of during the first
	 *  execution.
	 *  At this point, the clfft runtime applies all implimented optimizations, including
	 *  running kernel experiments on the devices in the plan context.
	 *  <p>  This function takes a long time to execute. If a plan is not baked before being executed,
	 *  the first call to clfftEnqueueTransform takes a long time to execute.
	 *  <p>  If any significant parameter of a plan is changed after the plan is baked (by a subsequent call to any one of
	 *  the functions that has the prefix "clfftSetPlan"), it is not considered an error.  Instead, the plan reverts back to
	 *  the unbaked state, discarding the benefits of the baking operation.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] numQueues Number of command queues in commQueueFFT; 0 is a valid value, in which case the client does not want
	 * 	the runtime to run load experiments and only pre-calculate state information
	 *  @param[in] commQueueFFT An array of cl_command_queues created by the client; the command queues must be a proper subset of
	 * 	the devices included in the plan context
	 *  @param[in] pfn_notify A function pointer to a notification routine. The notification routine is a callback function that
	 *  an application can register and is called when the program executable is built (successfully or unsuccessfully).
	 *  Currently, this parameter MUST be NULL or nullptr.
	 *  @param[in] user_data Passed as an argument when pfn_notify is called.
	 *  Currently, this parameter MUST be NULL or nullptr.
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftBakePlan(plHandle: clfftPlanHandle, numQueues: cl_uint,
                         commQueueFFT: *mut cl_command_queue,
                         pfn_notify:
                             ::std::option::Option<unsafe extern "C" fn(plHandle:
                                                                            clfftPlanHandle,
                                                                        user_data:
                                                                            *mut ::std::os::raw::c_void)>,
                         user_data: *mut ::std::os::raw::c_void)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Release the resources of a plan.
	 *  @details A plan may include resources, such as kernels, programs, and buffers that consume memory.  When a plan
	 *  is no more needed, the client must release the plan.
	 *  @param[in,out] plHandle Handle to a previously created plan
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftDestroyPlan(plHandle: *mut clfftPlanHandle) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the OpenCL context of a previously created plan.
	 *  @details The user must pass a reference to a cl_context variable, which is modified to point to a
	 *  context set in the specified plan.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] context Reference to the user allocated cl_context, which points to context set in the plan
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftGetPlanContext(plHandle: clfftPlanHandle,
                               context: *mut cl_context) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the floating point precision of the FFT data
	 *  @details The user must pass a reference to a clfftPrecision variable, which is set to the
	 *  precision of the FFT complex data in the plan.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] precision Reference to the user clfftPrecision enum
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftGetPlanPrecision(plHandle: clfftPlanHandle,
                                 precision: *mut clfftPrecision)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Set the floating point precision of the FFT data
	 *  @details Sets the floating point precision of the FFT complex data in the plan.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] precision Reference to the user clfftPrecision enum
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftSetPlanPrecision(plHandle: clfftPlanHandle,
                                 precision: clfftPrecision) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the scaling factor that is applied to the FFT data
	 *  @details The user must pass a reference to a cl_float variable, which is set to the
	 *  floating point scaling factor that is multiplied across the FFT data.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dir Direction of the applied scaling factor
	 *  @param[out] scale Reference to the user cl_float variable
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftGetPlanScale(plHandle: clfftPlanHandle, dir: clfftDirection,
                             scale: *mut cl_float) -> clfftStatus;
}
extern "C" {
    /*  @brief Set the scaling factor that is applied to the FFT data
	 *  @details Sets the floating point scaling factor that is
	 *  multiplied across the FFT data.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dir Direction of the applied scaling factor
	 *  @param[in] scale Reference to the user cl_float variable
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftSetPlanScale(plHandle: clfftPlanHandle, dir: clfftDirection,
                             scale: cl_float) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the number of discrete arrays that the plan can concurrently handle
	 *  @details The user must pass a reference to a cl_uint variable, which is set to the
	 *  number of discrete arrays (1D or 2D) that is batched together for the plan
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] batchSize Number of discrete FFTs performed
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftGetPlanBatchSize(plHandle: clfftPlanHandle,
                                 batchSize: *mut usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Set the number of discrete arrays that the plan can concurrently handle
	 *  @details Sets the plan property which sets the number of discrete arrays (1D or 2D)
	 *  that is batched together for the plan
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] batchSize Number of discrete FFTs performed
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftSetPlanBatchSize(plHandle: clfftPlanHandle, batchSize: usize)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the dimensionality of the data that is transformed
	 *  @details Queries a plan object and retrieves the value of the dimensionality that the plan is set for.  A size is returned to
	 *  help the client allocate sufficient storage to hold the dimensions in a further call to clfftGetPlanLength
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] dim The dimensionality of the FFT to be transformed
	 *  @param[out] size Value to allocate an array to hold the FFT dimensions.
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftGetPlanDim(plHandle: clfftPlanHandle, dim: *mut clfftDim,
                           size: *mut cl_uint) -> clfftStatus;
}
extern "C" {
    /*  @brief Set the dimensionality of the data that is transformed
	 *  @details Set the dimensionality of the data that is transformed by the plan
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dim The dimensionality of the FFT to be transformed
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftSetPlanDim(plHandle: clfftPlanHandle, dim: clfftDim)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the length of each dimension of the FFT
	 *  @details The user must pass a reference to a size_t array, which is set to the
	 *  length of each discrete dimension of the FFT
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dim Dimension of the FFT; describes how many elements are in the clLengths array
	 *  @param[out] clLengths An array of length of size 'dim';  each array value describes the length of each dimension
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftGetPlanLength(plHandle: clfftPlanHandle, dim: clfftDim,
                              clLengths: *mut usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Set the length of each dimension of the FFT
	 *  @details Sets the plan property which is the length of each discrete dimension of the FFT
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dim The dimension of the FFT; describes how many elements are in the clLengths array
	 *  @param[in] clLengths An array of length of size 'dim';  each array value describes the length of each dimension
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftSetPlanLength(plHandle: clfftPlanHandle, dim: clfftDim,
                              clLengths: *const usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the distance between consecutive elements of input buffers in each dimension.
	 *  @details Depending on how the dimension is set in the plan (for 2D or 3D FFT), strideY or strideZ can be safely
	 *  ignored
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dim The dimension of the stride parameters; provides the number of elements in the array
	 *  @param[out] clStrides An array of strides, of size 'dim'.
	 */
    pub fn clfftGetPlanInStride(plHandle: clfftPlanHandle, dim: clfftDim,
                                clStrides: *mut usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Set the distance between consecutive elements of input buffers in each dimension.
	 *  @details Set the plan properties which is the distance between elements in all dimensions of the input buffer
	 *  (units are in terms of clfftPrecision)
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dim The dimension of the stride parameters; provides the number of elements in the clStrides array
	 *  @param[in] clStrides An array of strides of size 'dim'. Usually, strideX=1 so that successive elements in the first dimension are stored contiguously.
	 * 	Typically, strideY=LenX and strideZ=LenX*LenY with the successive elements in the second and third dimensions stored in packed format.
	 *  See  @ref DistanceStridesandPitches for details.
	 */
    pub fn clfftSetPlanInStride(plHandle: clfftPlanHandle, dim: clfftDim,
                                clStrides: *mut usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the distance between consecutive elements of output buffers in each dimension.
	 *  @details Depending on how the dimension is set in the plan (for 2D or 3D FFT), strideY or strideZ can be safely
	 *  ignored
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dim The dimension of the stride parameters; provides the number of elements in the clStrides array
	 *  @param[out] clStrides An array of strides, of size 'dim'.
	 */
    pub fn clfftGetPlanOutStride(plHandle: clfftPlanHandle, dim: clfftDim,
                                 clStrides: *mut usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Set the distance between consecutive elements of output buffers in a dimension.
	 *  @details Sets the plan properties which is the distance between elements in all dimensions of the output buffer
	 *  (units are in terms of clfftPrecision)
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dim The dimension of the stride parameters; provides the number of elements in the clStrides array
	 *  @param[in] clStrides An array of strides of size 'dim'.  Usually, strideX=1 so that successive elements in the first dimension are stored contiguously.
	 * 	Typically, strideY=LenX and strideZ=LenX*LenY cause the successive elements in the second and third dimensions be stored in packed format.
	 *  @sa clfftSetPlanInStride
	 */
    pub fn clfftSetPlanOutStride(plHandle: clfftPlanHandle, dim: clfftDim,
                                 clStrides: *mut usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the distance between array objects
	 *  @details Pitch is the distance between each discrete array object in an FFT array. This is only used
	 *  for 'array' dimensions in clfftDim; see clfftSetPlanDimension (units are in terms of clfftPrecision)
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] iDist The distance between the beginning elements of the discrete array objects in input buffer.
	 *  For contiguous arrays in memory, iDist=(strideX*strideY*strideZ)
	 *  @param[out] oDist The distance between the beginning elements of the discrete array objects in output buffer.
	 *  For contiguous arrays in memory, oDist=(strideX*strideY*strideZ)
	 */
    pub fn clfftGetPlanDistance(plHandle: clfftPlanHandle, iDist: *mut usize,
                                oDist: *mut usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Set the distance between array objects
	 *  @details Pitch is the distance between each discrete array object in an FFT array. This is only used
	 *  for 'array' dimensions in clfftDim; see clfftSetPlanDimension (units are in terms of clfftPrecision)
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] iDist The distance between the beginning elements of the discrete array objects in input buffer.
	 *  For contiguous arrays in memory, iDist=(strideX*strideY*strideZ)
	 *  @param[out] oDist The distance between the beginning elements of the discrete array objects in output buffer.
	 *  For contiguous arrays in memory, oDist=(strideX*strideY*strideZ)
	 */
    pub fn clfftSetPlanDistance(plHandle: clfftPlanHandle, iDist: usize,
                                oDist: usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the expected layout of the input and output buffers
	 *  @details Input and output buffers can be filled with either Hermitian, complex, or real numbers.  Complex numbers are stored
	 *  in various layouts; this function retrieves the layouts used by input and output
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] iLayout Indicates how the input buffers are laid out in memory
	 *  @param[out] oLayout Indicates how the output buffers are laid out in memory
	 */
    pub fn clfftGetLayout(plHandle: clfftPlanHandle,
                          iLayout: *mut clfftLayout,
                          oLayout: *mut clfftLayout) -> clfftStatus;
}
extern "C" {
    /*  @brief Set the expected layout of the input and output buffers
	 *  @details Input and output buffers can be filled with either Hermitian, complex, or real numbers.  Complex numbers can be stored
	 *  in various layouts; this function informs the library what layouts to use for input and output
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] iLayout Indicates how the input buffers are laid out in memory
	 *  @param[in] oLayout Indicates how the output buffers are laid out in memory
	 */
    pub fn clfftSetLayout(plHandle: clfftPlanHandle, iLayout: clfftLayout,
                          oLayout: clfftLayout) -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve whether the input buffers are to be overwritten with results
	 *  @details If the setting performs an in-place transform, the input buffers are overwritten with the results of the
	 *  transform.  If the setting performs an out-of-place transforms, the library looks for separate output buffers
	 *  on the Enqueue call.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] placeness Informs the library to either overwrite the input buffers with results or to write them in separate output buffers
	 */
    pub fn clfftGetResultLocation(plHandle: clfftPlanHandle,
                                  placeness: *mut clfftResultLocation)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Set whether the input buffers are to be overwritten with results
	 *  @details If the setting performs an in-place transform, the input buffers are overwritten with the results of the
	 *  transform.  If the setting performs an out-of-place transforms, the library looks for separate output buffers
	 *  on the Enqueue call.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] placeness Informs the library to either overwrite the input buffers with results or to write them in separate output buffers
	 */
    pub fn clfftSetResultLocation(plHandle: clfftPlanHandle,
                                  placeness: clfftResultLocation)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Retrieve the final transpose setting of a multi-dimensional FFT
	 *  @details A multi-dimensional FFT transposes the data several times during calculation. If the client
	 *  does not care about the final transpose, to put data back in proper dimension, the final transpose can be skipped
	 *  to improve speed
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] transposed Specifies whether the final transpose can be skipped
	 */
    pub fn clfftGetPlanTransposeResult(plHandle: clfftPlanHandle,
                                       transposed: *mut clfftResultTransposed)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Set the final transpose setting of a multi-dimensional FFT
	 *  @details A multi-dimensional FFT transposes the data several times during calculation.  If the client
	 *  does not care about the final transpose, to put data back in proper dimension, the final transpose can be skipped
	 *  to improve speed
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] transposed Specifies whether the final transpose can be skipped
	 */
    pub fn clfftSetPlanTransposeResult(plHandle: clfftPlanHandle,
                                       transposed: clfftResultTransposed)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Get buffer size (in bytes), which may be needed internally for an intermediate buffer
	 *  @details Very large FFT transforms may need multiple passes, and the operation needs a temporary buffer to hold
	 *  intermediate results. This function is only valid after the plan is baked, otherwise, an invalid operation error
	 *  is returned. If the returned buffersize is 0, the runtime needs no temporary buffer.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[out] buffersize Size in bytes for intermediate buffer
	 */
    pub fn clfftGetTmpBufSize(plHandle: clfftPlanHandle,
                              buffersize: *mut usize) -> clfftStatus;
}
extern "C" {
    /*  @brief Register the callback parameters
	 *  @details Client can provide a callback function to do custom processing while reading input data and/or
	 *  writing output data. The callback function is provided as a string.
	 *  clFFT library incorporates the callback function string into the main FFT kernel. This function is used
	 *  by client to set the necessary parameters for callback
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] funcName Callback function name
	 *  @param[in] funcString Callback function in string form
	 *  @param[in] localMemSize Optional - Size (bytes) of the local memory used by callback function; pass 0 if no local memory is used
	 *  @param[in] callbackType Type of callback - Pre-Callback or Post-Callback
	 *  @param[in] userdata Supplementary data if any used by callback function
	 *  @param[in] numUserdataBuffers Number of userdata buffers
	 */
    pub fn clfftSetPlanCallback(plHandle: clfftPlanHandle,
                                funcName: *const ::std::os::raw::c_char,
                                funcString: *const ::std::os::raw::c_char,
                                localMemSize: ::std::os::raw::c_int,
                                callbackType: clfftCallbackType,
                                userdata: *mut cl_mem,
                                numUserdataBuffers: ::std::os::raw::c_int)
     -> clfftStatus;
}
extern "C" {
    /*  @brief Enqueue an FFT transform operation, and return immediately (non-blocking)
	 *  @details This transform API function computes the FFT transform. It is non-blocking as it
	 *  only enqueues the OpenCL kernels for execution. The synchronization step must be managed by the user.
	 *  @param[in] plHandle Handle to a previously created plan
	 *  @param[in] dir Forward or backward transform
	 *  @param[in] numQueuesAndEvents Number of command queues in commQueues; number of expected events to be returned in outEvents
	 *  @param[in] commQueues An array of cl_command_queues created by the client; the command queues must be a proper subset of
	 * 	the devices included in the OpenCL context associated with the plan
	 *  @param[in] numWaitEvents Specify the number of elements in the eventWaitList array
	 *  @param[in] waitEvents Events for which the transform waits to complete before executing on the device
	 *  @param[out] outEvents The runtime fills this array with events corresponding one to one with the input command queues passed
	 *	in commQueues.  This parameter can have the value NULL or nullptr. When the value is NULL, the client is not interested in receiving notifications
	 *	when transforms are finished, otherwise, (if not NULL) the client is responsible for allocating this array with at least
	 *	as many elements as specified in numQueuesAndEvents.
	 *  @param[in] inputBuffers An array of cl_mem objects that contain data for processing by the FFT runtime. If the transform
	 *  is in-place, the FFT results overwrite the input buffers
	 *  @param[out] outputBuffers An array of cl_mem objects that store the results of out-of-place transforms. If the transform
	 *  is in-place, this parameter may be NULL or nullptr and is completely ignored
	 *  @param[in] tmpBuffer A cl_mem object that is reserved as a temporary buffer for FFT processing. If clTmpBuffers is NULL or nullptr,
	 *  and the library needs temporary storage, an internal temporary buffer is created on the fly managed by the library.
	 *  @return Enum describing error condition; superset of OpenCL error codes
	 */
    pub fn clfftEnqueueTransform(plHandle: clfftPlanHandle,
                                 dir: clfftDirection,
                                 numQueuesAndEvents: cl_uint,
                                 commQueues: *mut cl_command_queue,
                                 numWaitEvents: cl_uint,
                                 waitEvents: *const cl_event,
                                 outEvents: *mut cl_event,
                                 inputBuffers: *mut cl_mem,
                                 outputBuffers: *mut cl_mem,
                                 tmpBuffer: cl_mem) -> clfftStatus;
}
