initSidebarItems({"enum":[["Error","An enum containing either a `String` or one of several other error types."]],"fn":[["err","Creates a \"leaf future\" from an immediate value of a failed computation."],["ok","Creates a \"leaf future\" from an immediate value of a finished and successful computation."],["result","Creates a new \"leaf future\" which will resolve with the given result."]],"struct":[["FutureMemMap","A future which resolves to a `MemMap` as soon as its creating command completes."],["FutureRwGuard","A future that resolves to a read or write guard after ensuring that the data being guarded is appropriately locked during the execution of an OpenCL command."],["MemMap","A view of memory mapped by `clEnqueueMap{...}`."],["ReadGuard","Allows access to the data contained within a lock just like a mutex guard."],["RwVec","A locking `Vec` which interoperates with OpenCL events and Rust futures to provide exclusive access to data."],["WriteGuard","Allows access to the data contained within just like a mutex guard."]],"type":[["FutureReader",""],["FutureResult",""],["FutureWriter",""],["Result",""]]});