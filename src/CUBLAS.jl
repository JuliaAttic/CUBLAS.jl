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
    throw(statusmessage(status))
end

# find the cublas library
const libcublas = find_library(["libcublas"], ["/usr/local/cuda"])
if isempty(libcublas)
    error("CUBLAS library cannot be found")
end

include("libcublas.jl")

handle = cublasHandle_t[0]
cublasCreate_v2(handle)

A = ones(5,5)
B = ones(5,5)
d_A = CudaArray(A)
d_B = CudaArray(B)
d_C = CudaArray(Float64,(5,5))

# execute dgemm
# cublasDgemm_v2(cublasHandle_t handle,
#                cublasOperation_t transa,
#                cublasOperation_t transb,
#                int m,
#                int n,
#                int k,
#                const double *alpha, /* host or device pointer */
#                const double *A,
#                int lda,
#                const double *B,
#                int ldb,
#                const double *beta, /* host or device pointer */
#                double *C,
#                int ldc);
cublasDgemm_v2(handle[1],
               CUBLAS_OP_N,CUBLAS_OP_N,
               5,5,5,[1.0],
               d_A,5,
               d_B,5,
               [0.0],d_C,5)

# clean up cublas, must pass the handle!!!
cublasDestroy_v2(handle[1])

# copy result back to host
C = to_host(d_C)
show(C)

end # module
