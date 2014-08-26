# begin enum cublasStatus_t
typealias cublasStatus_t Uint32
const CUBLAS_STATUS_SUCCESS = 0
const CUBLAS_STATUS_NOT_INITIALIZED = 1
const CUBLAS_STATUS_ALLOC_FAILED = 3
const CUBLAS_STATUS_INVALID_VALUE = 7
const CUBLAS_STATUS_ARCH_MISMATCH = 8
const CUBLAS_STATUS_MAPPING_ERROR = 11
const CUBLAS_STATUS_EXECUTION_FAILED = 13
const CUBLAS_STATUS_INTERNAL_ERROR = 14
const CUBLAS_STATUS_NOT_SUPPORTED = 15
const CUBLAS_STATUS_LICENSE_ERROR = 16
# end enum cublasStatus_t
# begin enum cublasFillMode_t
typealias cublasFillMode_t Uint32
const CUBLAS_FILL_MODE_LOWER = 0
const CUBLAS_FILL_MODE_UPPER = 1
# end enum cublasFillMode_t
# begin enum cublasDiagType_t
typealias cublasDiagType_t Uint32
const CUBLAS_DIAG_NON_UNIT = 0
const CUBLAS_DIAG_UNIT = 1
# end enum cublasDiagType_t
# begin enum cublasSideMode_t
typealias cublasSideMode_t Uint32
const CUBLAS_SIDE_LEFT = 0
const CUBLAS_SIDE_RIGHT = 1
# end enum cublasSideMode_t
# begin enum cublasOperation_t
typealias cublasOperation_t Uint32
const CUBLAS_OP_N = 0
const CUBLAS_OP_T = 1
const CUBLAS_OP_C = 2
# end enum cublasOperation_t
# begin enum cublasPointerMode_t
typealias cublasPointerMode_t Uint32
const CUBLAS_POINTER_MODE_HOST = 0
const CUBLAS_POINTER_MODE_DEVICE = 1
# end enum cublasPointerMode_t
# begin enum cublasAtomicsMode_t
typealias cublasAtomicsMode_t Uint32
const CUBLAS_ATOMICS_NOT_ALLOWED = 0
const CUBLAS_ATOMICS_ALLOWED = 1
# end enum cublasAtomicsMode_t
typealias cublasHandle_t Ptr{cublasContext}
