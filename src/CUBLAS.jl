# CUBLAS.jl
#
# Julia interface to CUBLAS.  Inspired by CUDArt.jl and using Clang.jl.
#
# Author: Nick Henderson <nwh@stanford.edu>
# Created: 2014-08-27
# License: MIT
#

module CUBLAS

using CUDArt
#using CUDArt.CudaPtr

import Base.LinAlg.BlasChar
import Base.one
import Base.zero

export band, unband

include("util.jl")
include("libcublas_types.jl")

# get cublas status message
function statusmessage(status)
    if status == CUBLAS_STATUS_SUCCESS
        return "cublas success"
    end
    if status == CUBLAS_STATUS_NOT_INITIALIZED
        return "cublas not initialized"
    end
    if status == CUBLAS_STATUS_ALLOC_FAILED
        return "cublas alloc failed"
    end
    if status == CUBLAS_STATUS_INVALID_VALUE
        return "cublas invalid value"
    end
    if status == CUBLAS_STATUS_ARCH_MISMATCH
        return "cublas arch mismatch"
    end
    if status == CUBLAS_STATUS_MAPPING_ERROR
        return "cublas mapping error"
    end
    if status == CUBLAS_STATUS_EXECUTION_FAILED
        return "cublas execution failed"
    end
    if status == CUBLAS_STATUS_INTERNAL_ERROR
        return "cublas internal error"
    end
    if status == CUBLAS_STATUS_NOT_SUPPORTED
        return "cublas not supported"
    end
    if status == CUBLAS_STATUS_LICENSE_ERROR
        return "cublas license error"
    end
    return "cublas unknown status"
end

# error handling function
function statuscheck(status)
    if status == CUBLAS_STATUS_SUCCESS
        return nothing
    end
    # Because try/finally may disguise the source of the problem,
    # let's show a backtrace here
    warn("CUBLAS error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    println()
    throw(statusmessage(status))
end

# find the cublas library
const libcublas = find_library(["libcublas"], ["/usr/local/cuda"])
if isempty(libcublas)
    error("CUBLAS library cannot be found")
end

include("libcublas.jl")

# setup cublas handle
cublashandle = cublasHandle_t[0]
cublasCreate_v2(cublashandle)
# destroy cublas handle at julia exit
atexit(()->cublasDestroy_v2(cublashandle[1]))

include("blas.jl")

end # module
