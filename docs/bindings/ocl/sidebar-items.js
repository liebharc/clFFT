initSidebarItems({"enum":[["Error","An enum containing either a `String` or one of several other error types."],["SpatialDims","Specifies a size or offset in up to three dimensions."]],"mod":[["async","Types related to futures and asynchrony."],["builders","Builders and associated settings-related types."],["enums","Enumerators for settings and information requests."],["flags","Bitflags for various parameter types."],["prm","OpenCL scalar and vector primitive types."],["traits","Commonly used traits."]],"struct":[["Buffer","A chunk of memory physically located on a device, such as a GPU."],["Context","A context for a particular platform and set of device types."],["Device","An individual device identifier (an OpenCL device_id)."],["Event","An event representing a command or user created event."],["EventArray","A list of events for coordinating enqueued commands."],["EventList","A list of events for coordinating enqueued commands."],["FutureMemMap","A future which resolves to a `MemMap` as soon as its creating command completes."],["FutureRwGuard","A future that resolves to a read or write guard after ensuring that the data being guarded is appropriately locked during the execution of an OpenCL command."],["Image","A section of device memory which represents one or many images."],["Kernel","A kernel which represents a 'procedure'."],["MemMap","A view of memory mapped by `clEnqueueMap{...}`."],["Platform","A platform identifier."],["ProQue","An all-in-one chimera of the `Program`, `Queue`, `Context` and (optionally) `SpatialDims` types."],["Program","A program from which kernels can be created from."],["Queue","A command queue which manages all actions taken on kernels, buffers, and images."],["ReadGuard","Allows access to the data contained within a lock just like a mutex guard."],["RwVec","A locking `Vec` which interoperates with OpenCL events and Rust futures to provide exclusive access to data."],["Sampler","An image sampler used to process images."],["WriteGuard","Allows access to the data contained within just like a mutex guard."]],"type":[["FutureReader",""],["FutureWriter",""],["Result","`ocl::Error` result type."]]});