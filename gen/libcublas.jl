# Julia wrapper for header: /usr/local/cuda/include/cublas_v2.h
# Automatically generated using Clang.jl wrap_c, version 0.0.0

function cublasCreate_v2(handle::Ptr{cublasHandle_t})
  ccall( (:cublasCreate_v2, libcublas), cublasStatus_t, (Ptr{cublasHandle_t},), handle)
end
function cublasDestroy_v2(handle::cublasHandle_t)
  ccall( (:cublasDestroy_v2, libcublas), cublasStatus_t, (cublasHandle_t,), handle)
end
function cublasGetVersion_v2(handle::cublasHandle_t, version::Ptr{Cint})
  ccall( (:cublasGetVersion_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cint}), handle, version)
end
function cublasSetStream_v2(handle::cublasHandle_t, streamId::cudaStream_t)
  ccall( (:cublasSetStream_v2, libcublas), cublasStatus_t, (cublasHandle_t, cudaStream_t), handle, streamId)
end
function cublasGetStream_v2(handle::cublasHandle_t, streamId::Ptr{cudaStream_t})
  ccall( (:cublasGetStream_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cudaStream_t}), handle, streamId)
end
function cublasGetPointerMode_v2(handle::cublasHandle_t, mode::Ptr{cublasPointerMode_t})
  ccall( (:cublasGetPointerMode_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cublasPointerMode_t}), handle, mode)
end
function cublasSetPointerMode_v2(handle::cublasHandle_t, mode::cublasPointerMode_t)
  ccall( (:cublasSetPointerMode_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasPointerMode_t), handle, mode)
end
function cublasGetAtomicsMode(handle::cublasHandle_t, mode::Ptr{cublasAtomicsMode_t})
  ccall( (:cublasGetAtomicsMode, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cublasAtomicsMode_t}), handle, mode)
end
function cublasSetAtomicsMode(handle::cublasHandle_t, mode::cublasAtomicsMode_t)
  ccall( (:cublasSetAtomicsMode, libcublas), cublasStatus_t, (cublasHandle_t, cublasAtomicsMode_t), handle, mode)
end
function cublasSetVector(n::Cint, elemSize::Cint, x::Ptr{None}, incx::Cint, devicePtr::Ptr{None}, incy::Cint)
  ccall( (:cublasSetVector, libcublas), cublasStatus_t, (Cint, Cint, Ptr{None}, Cint, Ptr{None}, Cint), n, elemSize, x, incx, devicePtr, incy)
end
function cublasGetVector(n::Cint, elemSize::Cint, x::Ptr{None}, incx::Cint, y::Ptr{None}, incy::Cint)
  ccall( (:cublasGetVector, libcublas), cublasStatus_t, (Cint, Cint, Ptr{None}, Cint, Ptr{None}, Cint), n, elemSize, x, incx, y, incy)
end
function cublasSetMatrix(rows::Cint, cols::Cint, elemSize::Cint, A::Ptr{None}, lda::Cint, B::Ptr{None}, ldb::Cint)
  ccall( (:cublasSetMatrix, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{None}, Cint, Ptr{None}, Cint), rows, cols, elemSize, A, lda, B, ldb)
end
function cublasGetMatrix(rows::Cint, cols::Cint, elemSize::Cint, A::Ptr{None}, lda::Cint, B::Ptr{None}, ldb::Cint)
  ccall( (:cublasGetMatrix, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{None}, Cint, Ptr{None}, Cint), rows, cols, elemSize, A, lda, B, ldb)
end
function cublasSetVectorAsync(n::Cint, elemSize::Cint, hostPtr::Ptr{None}, incx::Cint, devicePtr::Ptr{None}, incy::Cint, stream::cudaStream_t)
  ccall( (:cublasSetVectorAsync, libcublas), cublasStatus_t, (Cint, Cint, Ptr{None}, Cint, Ptr{None}, Cint, cudaStream_t), n, elemSize, hostPtr, incx, devicePtr, incy, stream)
end
function cublasGetVectorAsync(n::Cint, elemSize::Cint, devicePtr::Ptr{None}, incx::Cint, hostPtr::Ptr{None}, incy::Cint, stream::cudaStream_t)
  ccall( (:cublasGetVectorAsync, libcublas), cublasStatus_t, (Cint, Cint, Ptr{None}, Cint, Ptr{None}, Cint, cudaStream_t), n, elemSize, devicePtr, incx, hostPtr, incy, stream)
end
function cublasSetMatrixAsync(rows::Cint, cols::Cint, elemSize::Cint, A::Ptr{None}, lda::Cint, B::Ptr{None}, ldb::Cint, stream::cudaStream_t)
  ccall( (:cublasSetMatrixAsync, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{None}, Cint, Ptr{None}, Cint, cudaStream_t), rows, cols, elemSize, A, lda, B, ldb, stream)
end
function cublasGetMatrixAsync(rows::Cint, cols::Cint, elemSize::Cint, A::Ptr{None}, lda::Cint, B::Ptr{None}, ldb::Cint, stream::cudaStream_t)
  ccall( (:cublasGetMatrixAsync, libcublas), cublasStatus_t, (Cint, Cint, Cint, Ptr{None}, Cint, Ptr{None}, Cint, cudaStream_t), rows, cols, elemSize, A, lda, B, ldb, stream)
end
function cublasXerbla(srName::Ptr{Uint8}, info::Cint)
  ccall( (:cublasXerbla, libcublas), None, (Ptr{Uint8}, Cint), srName, info)
end
function cublasSnrm2_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, result::Ptr{Cfloat})
  ccall( (:cublasSnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, n, x, incx, result)
end
function cublasDnrm2_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, result::Ptr{Cdouble})
  ccall( (:cublasDnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, n, x, incx, result)
end
function cublasScnrm2_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, result::Ptr{Cfloat})
  ccall( (:cublasScnrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}), handle, n, x, incx, result)
end
function cublasDznrm2_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, result::Ptr{Cdouble})
  ccall( (:cublasDznrm2_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}), handle, n, x, incx, result)
end
function cublasSdot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, result::Ptr{Cfloat})
  ccall( (:cublasSdot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, n, x, incx, y, incy, result)
end
function cublasDdot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, result::Ptr{Cdouble})
  ccall( (:cublasDdot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, n, x, incx, y, incy, result)
end
function cublasCdotu_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, result::Ptr{cuComplex})
  ccall( (:cublasCdotu_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, n, x, incx, y, incy, result)
end
function cublasCdotc_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, result::Ptr{cuComplex})
  ccall( (:cublasCdotc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, n, x, incx, y, incy, result)
end
function cublasZdotu_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, result::Ptr{cuDoubleComplex})
  ccall( (:cublasZdotu_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, n, x, incx, y, incy, result)
end
function cublasZdotc_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, result::Ptr{cuDoubleComplex})
  ccall( (:cublasZdotc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, n, x, incx, y, incy, result)
end
function cublasSscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)
  ccall( (:cublasSscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, n, alpha, x, incx)
end
function cublasDscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)
  ccall( (:cublasDscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, n, alpha, x, incx)
end
function cublasCscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint)
  ccall( (:cublasCscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, n, alpha, x, incx)
end
function cublasCsscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{cuComplex}, incx::Cint)
  ccall( (:cublasCsscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint), handle, n, alpha, x, incx)
end
function cublasZscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint)
  ccall( (:cublasZscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, n, alpha, x, incx)
end
function cublasZdscal_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{cuDoubleComplex}, incx::Cint)
  ccall( (:cublasZdscal_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint), handle, n, alpha, x, incx)
end
function cublasSaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)
  ccall( (:cublasSaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, n, alpha, x, incx, y, incy)
end
function cublasDaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)
  ccall( (:cublasDaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, n, alpha, x, incx, y, incy)
end
function cublasCaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasCaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, n, alpha, x, incx, y, incy)
end
function cublasZaxpy_v2(handle::cublasHandle_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZaxpy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, n, alpha, x, incx, y, incy)
end
function cublasScopy_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)
  ccall( (:cublasScopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, n, x, incx, y, incy)
end
function cublasDcopy_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)
  ccall( (:cublasDcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, n, x, incx, y, incy)
end
function cublasCcopy_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasCcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, n, x, incx, y, incy)
end
function cublasZcopy_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZcopy_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, n, x, incx, y, incy)
end
function cublasSswap_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint)
  ccall( (:cublasSswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, n, x, incx, y, incy)
end
function cublasDswap_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint)
  ccall( (:cublasDswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, n, x, incx, y, incy)
end
function cublasCswap_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasCswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, n, x, incx, y, incy)
end
function cublasZswap_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZswap_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, n, x, incx, y, incy)
end
function cublasIsamax_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, result::Ptr{Cint})
  ccall( (:cublasIsamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end
function cublasIdamax_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, result::Ptr{Cint})
  ccall( (:cublasIdamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end
function cublasIcamax_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, result::Ptr{Cint})
  ccall( (:cublasIcamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end
function cublasIzamax_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, result::Ptr{Cint})
  ccall( (:cublasIzamax_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end
function cublasIsamin_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, result::Ptr{Cint})
  ccall( (:cublasIsamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end
function cublasIdamin_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, result::Ptr{Cint})
  ccall( (:cublasIdamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end
function cublasIcamin_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, result::Ptr{Cint})
  ccall( (:cublasIcamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end
function cublasIzamin_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, result::Ptr{Cint})
  ccall( (:cublasIzamin_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cint}), handle, n, x, incx, result)
end
function cublasSasum_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, result::Ptr{Cfloat})
  ccall( (:cublasSasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, n, x, incx, result)
end
function cublasDasum_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, result::Ptr{Cdouble})
  ccall( (:cublasDasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, n, x, incx, result)
end
function cublasScasum_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, result::Ptr{Cfloat})
  ccall( (:cublasScasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}), handle, n, x, incx, result)
end
function cublasDzasum_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, result::Ptr{Cdouble})
  ccall( (:cublasDzasum_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}), handle, n, x, incx, result)
end
function cublasSrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, c::Ptr{Cfloat}, s::Ptr{Cfloat})
  ccall( (:cublasSrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}), handle, n, x, incx, y, incy, c, s)
end
function cublasDrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, c::Ptr{Cdouble}, s::Ptr{Cdouble})
  ccall( (:cublasDrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), handle, n, x, incx, y, incy, c, s)
end
function cublasCrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, c::Ptr{Cfloat}, s::Ptr{cuComplex})
  ccall( (:cublasCrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}), handle, n, x, incx, y, incy, c, s)
end
function cublasCsrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, c::Ptr{Cfloat}, s::Ptr{Cfloat})
  ccall( (:cublasCsrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{Cfloat}), handle, n, x, incx, y, incy, c, s)
end
function cublasZrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, c::Ptr{Cdouble}, s::Ptr{cuDoubleComplex})
  ccall( (:cublasZrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}), handle, n, x, incx, y, incy, c, s)
end
function cublasZdrot_v2(handle::cublasHandle_t, n::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, c::Ptr{Cdouble}, s::Ptr{Cdouble})
  ccall( (:cublasZdrot_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{Cdouble}), handle, n, x, incx, y, incy, c, s)
end
function cublasSrotg_v2(handle::cublasHandle_t, a::Ptr{Cfloat}, b::Ptr{Cfloat}, c::Ptr{Cfloat}, s::Ptr{Cfloat})
  ccall( (:cublasSrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), handle, a, b, c, s)
end
function cublasDrotg_v2(handle::cublasHandle_t, a::Ptr{Cdouble}, b::Ptr{Cdouble}, c::Ptr{Cdouble}, s::Ptr{Cdouble})
  ccall( (:cublasDrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), handle, a, b, c, s)
end
function cublasCrotg_v2(handle::cublasHandle_t, a::Ptr{cuComplex}, b::Ptr{cuComplex}, c::Ptr{Cfloat}, s::Ptr{cuComplex})
  ccall( (:cublasCrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{Cfloat}, Ptr{cuComplex}), handle, a, b, c, s)
end
function cublasZrotg_v2(handle::cublasHandle_t, a::Ptr{cuDoubleComplex}, b::Ptr{cuDoubleComplex}, c::Ptr{Cdouble}, s::Ptr{cuDoubleComplex})
  ccall( (:cublasZrotg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{Cdouble}, Ptr{cuDoubleComplex}), handle, a, b, c, s)
end
function cublasSrotm_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, param::Ptr{Cfloat})
  ccall( (:cublasSrotm_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, n, x, incx, y, incy, param)
end
function cublasDrotm_v2(handle::cublasHandle_t, n::Cint, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, param::Ptr{Cdouble})
  ccall( (:cublasDrotm_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, n, x, incx, y, incy, param)
end
function cublasSrotmg_v2(handle::cublasHandle_t, d1::Ptr{Cfloat}, d2::Ptr{Cfloat}, x1::Ptr{Cfloat}, y1::Ptr{Cfloat}, param::Ptr{Cfloat})
  ccall( (:cublasSrotmg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}), handle, d1, d2, x1, y1, param)
end
function cublasDrotmg_v2(handle::cublasHandle_t, d1::Ptr{Cdouble}, d2::Ptr{Cdouble}, x1::Ptr{Cdouble}, y1::Ptr{Cdouble}, param::Ptr{Cdouble})
  ccall( (:cublasDrotmg_v2, libcublas), cublasStatus_t, (cublasHandle_t, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), handle, d1, d2, x1, y1, param)
end
function cublasSgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)
  ccall( (:cublasSgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasDgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)
  ccall( (:cublasDgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasCgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasCgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasZgemv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZgemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasSgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)
  ccall( (:cublasSgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasDgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)
  ccall( (:cublasDgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasCgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasCgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasZgbmv_v2(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, kl::Cint, ku::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZgbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasStrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)
  ccall( (:cublasStrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end
function cublasDtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)
  ccall( (:cublasDtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end
function cublasCtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)
  ccall( (:cublasCtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end
function cublasZtrmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)
  ccall( (:cublasZtrmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end
function cublasStbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)
  ccall( (:cublasStbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end
function cublasDtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)
  ccall( (:cublasDtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end
function cublasCtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)
  ccall( (:cublasCtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end
function cublasZtbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)
  ccall( (:cublasZtbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end
function cublasStpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)
  ccall( (:cublasStpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end
function cublasDtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)
  ccall( (:cublasDtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end
function cublasCtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint)
  ccall( (:cublasCtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end
function cublasZtpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint)
  ccall( (:cublasZtpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end
function cublasStrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)
  ccall( (:cublasStrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end
function cublasDtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)
  ccall( (:cublasDtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end
function cublasCtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)
  ccall( (:cublasCtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end
function cublasZtrsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)
  ccall( (:cublasZtrsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, A, lda, x, incx)
end
function cublasStpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint)
  ccall( (:cublasStpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end
function cublasDtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint)
  ccall( (:cublasDtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end
function cublasCtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint)
  ccall( (:cublasCtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end
function cublasZtpsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint)
  ccall( (:cublasZtpsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, AP, x, incx)
end
function cublasStbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint)
  ccall( (:cublasStbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end
function cublasDtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint)
  ccall( (:cublasDtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end
function cublasCtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint)
  ccall( (:cublasCtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end
function cublasZtbsv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, n::Cint, k::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint)
  ccall( (:cublasZtbsv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, diag, n, k, A, lda, x, incx)
end
function cublasSsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)
  ccall( (:cublasSsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasDsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)
  ccall( (:cublasDsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasCsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasCsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasZsymv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZsymv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasChemv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasChemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasZhemv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZhemv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasSsbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)
  ccall( (:cublasSsbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasDsbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)
  ccall( (:cublasDsbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasChbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasChbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasZhbmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZhbmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
end
function cublasSspmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, AP::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, beta::Ptr{Cfloat}, y::Ptr{Cfloat}, incy::Cint)
  ccall( (:cublasSspmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end
function cublasDspmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, AP::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, beta::Ptr{Cdouble}, y::Ptr{Cdouble}, incy::Cint)
  ccall( (:cublasDspmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end
function cublasChpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, AP::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, beta::Ptr{cuComplex}, y::Ptr{cuComplex}, incy::Cint)
  ccall( (:cublasChpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end
function cublasZhpmv_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, AP::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, beta::Ptr{cuDoubleComplex}, y::Ptr{cuDoubleComplex}, incy::Cint)
  ccall( (:cublasZhpmv_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, AP, x, incx, beta, y, incy)
end
function cublasSger_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, A::Ptr{Cfloat}, lda::Cint)
  ccall( (:cublasSger_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end
function cublasDger_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, A::Ptr{Cdouble}, lda::Cint)
  ccall( (:cublasDger_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end
function cublasCgeru_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)
  ccall( (:cublasCgeru_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end
function cublasCgerc_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)
  ccall( (:cublasCgerc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end
function cublasZgeru_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)
  ccall( (:cublasZgeru_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end
function cublasZgerc_v2(handle::cublasHandle_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)
  ccall( (:cublasZgerc_v2, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, m, n, alpha, x, incx, y, incy, A, lda)
end
function cublasSsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, A::Ptr{Cfloat}, lda::Cint)
  ccall( (:cublasSsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end
function cublasDsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, A::Ptr{Cdouble}, lda::Cint)
  ccall( (:cublasDsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end
function cublasCsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, A::Ptr{cuComplex}, lda::Cint)
  ccall( (:cublasCsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end
function cublasZsyr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)
  ccall( (:cublasZsyr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end
function cublasCher_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{cuComplex}, incx::Cint, A::Ptr{cuComplex}, lda::Cint)
  ccall( (:cublasCher_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end
function cublasZher_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{cuDoubleComplex}, incx::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)
  ccall( (:cublasZher_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, x, incx, A, lda)
end
function cublasSspr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, AP::Ptr{Cfloat})
  ccall( (:cublasSspr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, uplo, n, alpha, x, incx, AP)
end
function cublasDspr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, AP::Ptr{Cdouble})
  ccall( (:cublasDspr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, uplo, n, alpha, x, incx, AP)
end
function cublasChpr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{cuComplex}, incx::Cint, AP::Ptr{cuComplex})
  ccall( (:cublasChpr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, uplo, n, alpha, x, incx, AP)
end
function cublasZhpr_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{cuDoubleComplex}, incx::Cint, AP::Ptr{cuDoubleComplex})
  ccall( (:cublasZhpr_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, uplo, n, alpha, x, incx, AP)
end
function cublasSsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, A::Ptr{Cfloat}, lda::Cint)
  ccall( (:cublasSsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end
function cublasDsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, A::Ptr{Cdouble}, lda::Cint)
  ccall( (:cublasDsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end
function cublasCsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)
  ccall( (:cublasCsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end
function cublasZsyr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)
  ccall( (:cublasZsyr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end
function cublasCher2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, A::Ptr{cuComplex}, lda::Cint)
  ccall( (:cublasCher2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end
function cublasZher2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, A::Ptr{cuDoubleComplex}, lda::Cint)
  ccall( (:cublasZher2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, alpha, x, incx, y, incy, A, lda)
end
function cublasSspr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cfloat}, x::Ptr{Cfloat}, incx::Cint, y::Ptr{Cfloat}, incy::Cint, AP::Ptr{Cfloat})
  ccall( (:cublasSspr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, uplo, n, alpha, x, incx, y, incy, AP)
end
function cublasDspr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{Cdouble}, x::Ptr{Cdouble}, incx::Cint, y::Ptr{Cdouble}, incy::Cint, AP::Ptr{Cdouble})
  ccall( (:cublasDspr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, uplo, n, alpha, x, incx, y, incy, AP)
end
function cublasChpr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuComplex}, x::Ptr{cuComplex}, incx::Cint, y::Ptr{cuComplex}, incy::Cint, AP::Ptr{cuComplex})
  ccall( (:cublasChpr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, uplo, n, alpha, x, incx, y, incy, AP)
end
function cublasZhpr2_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, alpha::Ptr{cuDoubleComplex}, x::Ptr{cuDoubleComplex}, incx::Cint, y::Ptr{cuDoubleComplex}, incy::Cint, AP::Ptr{cuDoubleComplex})
  ccall( (:cublasZhpr2_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, uplo, n, alpha, x, incx, y, incy, AP)
end
function cublasSgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)
  ccall( (:cublasSgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasDgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)
  ccall( (:cublasDgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasCgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasZgemm_v2(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZgemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasSsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)
  ccall( (:cublasSsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end
function cublasDsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)
  ccall( (:cublasDsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end
function cublasCsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end
function cublasZsyrk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZsyrk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end
function cublasCherk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{cuComplex}, lda::Cint, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCherk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end
function cublasZherk_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{cuDoubleComplex}, lda::Cint, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZherk_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
end
function cublasSsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)
  ccall( (:cublasSsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasDsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)
  ccall( (:cublasDsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasCsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasZsyr2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZsyr2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasCher2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCher2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasZher2k_v2(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZher2k_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasSsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)
  ccall( (:cublasSsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasDsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)
  ccall( (:cublasDsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasCsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasZsyrkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZsyrkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasCherkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCherkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{Cfloat}, Ptr{cuComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasZherkx(handle::cublasHandle_t, uplo::cublasFillMode_t, trans::cublasOperation_t, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZherkx, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{Cdouble}, Ptr{cuDoubleComplex}, Cint), handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasSsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, beta::Ptr{Cfloat}, C::Ptr{Cfloat}, ldc::Cint)
  ccall( (:cublasSsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasDsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, beta::Ptr{Cdouble}, C::Ptr{Cdouble}, ldc::Cint)
  ccall( (:cublasDsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasCsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasZsymm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZsymm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasChemm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, beta::Ptr{cuComplex}, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasChemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasZhemm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, beta::Ptr{cuDoubleComplex}, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZhemm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
end
function cublasStrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint)
  ccall( (:cublasStrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end
function cublasDtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint)
  ccall( (:cublasDtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end
function cublasCtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint)
  ccall( (:cublasCtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end
function cublasZtrsm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint)
  ccall( (:cublasZtrsm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb)
end
function cublasStrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, B::Ptr{Cfloat}, ldb::Cint, C::Ptr{Cfloat}, ldc::Cint)
  ccall( (:cublasStrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end
function cublasDtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, B::Ptr{Cdouble}, ldb::Cint, C::Ptr{Cdouble}, ldc::Cint)
  ccall( (:cublasDtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end
function cublasCtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, B::Ptr{cuComplex}, ldb::Cint, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end
function cublasZtrmm_v2(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, B::Ptr{cuDoubleComplex}, ldb::Cint, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZtrmm_v2, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
end
function cublasSgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cfloat}, Aarray::Ptr{Ptr{Cfloat}}, lda::Cint, Barray::Ptr{Ptr{Cfloat}}, ldb::Cint, beta::Ptr{Cfloat}, Carray::Ptr{Ptr{Cfloat}}, ldc::Cint, batchCount::Cint)
  ccall( (:cublasSgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end
function cublasDgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{Cdouble}, Aarray::Ptr{Ptr{Cdouble}}, lda::Cint, Barray::Ptr{Ptr{Cdouble}}, ldb::Cint, beta::Ptr{Cdouble}, Carray::Ptr{Ptr{Cdouble}}, ldc::Cint, batchCount::Cint)
  ccall( (:cublasDgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end
function cublasCgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuComplex}, Aarray::Ptr{Ptr{cuComplex}}, lda::Cint, Barray::Ptr{Ptr{cuComplex}}, ldb::Cint, beta::Ptr{cuComplex}, Carray::Ptr{Ptr{cuComplex}}, ldc::Cint, batchCount::Cint)
  ccall( (:cublasCgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end
function cublasZgemmBatched(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, k::Cint, alpha::Ptr{cuDoubleComplex}, Aarray::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, Barray::Ptr{Ptr{cuDoubleComplex}}, ldb::Cint, beta::Ptr{cuDoubleComplex}, Carray::Ptr{Ptr{cuDoubleComplex}}, ldc::Cint, batchCount::Cint)
  ccall( (:cublasZgemmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Cint), handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
end
function cublasSgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint, beta::Ptr{Cfloat}, B::Ptr{Cfloat}, ldb::Cint, C::Ptr{Cfloat}, ldc::Cint)
  ccall( (:cublasSgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end
function cublasDgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint, beta::Ptr{Cdouble}, B::Ptr{Cdouble}, ldb::Cint, C::Ptr{Cdouble}, ldc::Cint)
  ccall( (:cublasDgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end
function cublasCgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint, beta::Ptr{cuComplex}, B::Ptr{cuComplex}, ldb::Cint, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end
function cublasZgeam(handle::cublasHandle_t, transa::cublasOperation_t, transb::cublasOperation_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint, beta::Ptr{cuDoubleComplex}, B::Ptr{cuDoubleComplex}, ldb::Cint, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZgeam, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, cublasOperation_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
end
function cublasSgetrfBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cfloat}}, lda::Cint, P::Ptr{Cint}, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasSgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, n, A, lda, P, info, batchSize)
end
function cublasDgetrfBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cdouble}}, lda::Cint, P::Ptr{Cint}, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasDgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, n, A, lda, P, info, batchSize)
end
function cublasCgetrfBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuComplex}}, lda::Cint, P::Ptr{Cint}, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasCgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, n, A, lda, P, info, batchSize)
end
function cublasZgetrfBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, P::Ptr{Cint}, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasZgetrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, n, A, lda, P, info, batchSize)
end
function cublasSgetriBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cfloat}}, lda::Cint, P::Ptr{Cint}, C::Ptr{Ptr{Cfloat}}, ldc::Cint, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasSgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, P, C, ldc, info, batchSize)
end
function cublasDgetriBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cdouble}}, lda::Cint, P::Ptr{Cint}, C::Ptr{Ptr{Cdouble}}, ldc::Cint, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasDgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, P, C, ldc, info, batchSize)
end
function cublasCgetriBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuComplex}}, lda::Cint, P::Ptr{Cint}, C::Ptr{Ptr{cuComplex}}, ldc::Cint, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasCgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, P, C, ldc, info, batchSize)
end
function cublasZgetriBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, P::Ptr{Cint}, C::Ptr{Ptr{cuDoubleComplex}}, ldc::Cint, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasZgetriBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, P, C, ldc, info, batchSize)
end
function cublasStrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cfloat}, A::Ptr{Ptr{Cfloat}}, lda::Cint, B::Ptr{Ptr{Cfloat}}, ldb::Cint, batchCount::Cint)
  ccall( (:cublasStrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cfloat}, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end
function cublasDtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{Cdouble}, A::Ptr{Ptr{Cdouble}}, lda::Cint, B::Ptr{Ptr{Cdouble}}, ldb::Cint, batchCount::Cint)
  ccall( (:cublasDtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{Cdouble}, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end
function cublasCtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuComplex}, A::Ptr{Ptr{cuComplex}}, lda::Cint, B::Ptr{Ptr{cuComplex}}, ldb::Cint, batchCount::Cint)
  ccall( (:cublasCtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuComplex}, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end
function cublasZtrsmBatched(handle::cublasHandle_t, side::cublasSideMode_t, uplo::cublasFillMode_t, trans::cublasOperation_t, diag::cublasDiagType_t, m::Cint, n::Cint, alpha::Ptr{cuDoubleComplex}, A::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, B::Ptr{Ptr{cuDoubleComplex}}, ldb::Cint, batchCount::Cint)
  ccall( (:cublasZtrsmBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t, Cint, Cint, Ptr{cuDoubleComplex}, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Cint), handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount)
end
function cublasSmatinvBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cfloat}}, lda::Cint, Ainv::Ptr{Ptr{Cfloat}}, lda_inv::Cint, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasSmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end
function cublasDmatinvBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{Cdouble}}, lda::Cint, Ainv::Ptr{Ptr{Cdouble}}, lda_inv::Cint, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasDmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end
function cublasCmatinvBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuComplex}}, lda::Cint, Ainv::Ptr{Ptr{cuComplex}}, lda_inv::Cint, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasCmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end
function cublasZmatinvBatched(handle::cublasHandle_t, n::Cint, A::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, Ainv::Ptr{Ptr{cuDoubleComplex}}, lda_inv::Cint, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasZmatinvBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Cint), handle, n, A, lda, Ainv, lda_inv, info, batchSize)
end
function cublasSgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Ptr{Ptr{Cfloat}}, lda::Cint, TauArray::Ptr{Ptr{Cfloat}}, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasSgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Ptr{Cint}, Cint), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end
function cublasDgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Ptr{Ptr{Cdouble}}, lda::Cint, TauArray::Ptr{Ptr{Cdouble}}, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasDgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Ptr{Cint}, Cint), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end
function cublasCgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Ptr{Ptr{cuComplex}}, lda::Cint, TauArray::Ptr{Ptr{cuComplex}}, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasCgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Ptr{Cint}, Cint), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end
function cublasZgeqrfBatched(handle::cublasHandle_t, m::Cint, n::Cint, Aarray::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, TauArray::Ptr{Ptr{cuDoubleComplex}}, info::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasZgeqrfBatched, libcublas), cublasStatus_t, (cublasHandle_t, Cint, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Ptr{Cint}, Cint), handle, m, n, Aarray, lda, TauArray, info, batchSize)
end
function cublasSgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{Cfloat}}, lda::Cint, Carray::Ptr{Ptr{Cfloat}}, ldc::Cint, info::Ptr{Cint}, devInfoArray::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasSgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Ptr{Cfloat}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end
function cublasDgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{Cdouble}}, lda::Cint, Carray::Ptr{Ptr{Cdouble}}, ldc::Cint, info::Ptr{Cint}, devInfoArray::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasDgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Ptr{Cdouble}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end
function cublasCgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{cuComplex}}, lda::Cint, Carray::Ptr{Ptr{cuComplex}}, ldc::Cint, info::Ptr{Cint}, devInfoArray::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasCgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Ptr{cuComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end
function cublasZgelsBatched(handle::cublasHandle_t, trans::cublasOperation_t, m::Cint, n::Cint, nrhs::Cint, Aarray::Ptr{Ptr{cuDoubleComplex}}, lda::Cint, Carray::Ptr{Ptr{cuDoubleComplex}}, ldc::Cint, info::Ptr{Cint}, devInfoArray::Ptr{Cint}, batchSize::Cint)
  ccall( (:cublasZgelsBatched, libcublas), cublasStatus_t, (cublasHandle_t, cublasOperation_t, Cint, Cint, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Ptr{cuDoubleComplex}}, Cint, Ptr{Cint}, Ptr{Cint}, Cint), handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize)
end
function cublasSdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Ptr{Cfloat}, lda::Cint, x::Ptr{Cfloat}, incx::Cint, C::Ptr{Cfloat}, ldc::Cint)
  ccall( (:cublasSdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}, Cint), handle, mode, m, n, A, lda, x, incx, C, ldc)
end
function cublasDdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Ptr{Cdouble}, lda::Cint, x::Ptr{Cdouble}, incx::Cint, C::Ptr{Cdouble}, ldc::Cint)
  ccall( (:cublasDdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint), handle, mode, m, n, A, lda, x, incx, C, ldc)
end
function cublasCdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Ptr{cuComplex}, lda::Cint, x::Ptr{cuComplex}, incx::Cint, C::Ptr{cuComplex}, ldc::Cint)
  ccall( (:cublasCdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}, Cint), handle, mode, m, n, A, lda, x, incx, C, ldc)
end
function cublasZdgmm(handle::cublasHandle_t, mode::cublasSideMode_t, m::Cint, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, x::Ptr{cuDoubleComplex}, incx::Cint, C::Ptr{cuDoubleComplex}, ldc::Cint)
  ccall( (:cublasZdgmm, libcublas), cublasStatus_t, (cublasHandle_t, cublasSideMode_t, Cint, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}, Cint), handle, mode, m, n, A, lda, x, incx, C, ldc)
end
function cublasStpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Ptr{Cfloat}, A::Ptr{Cfloat}, lda::Cint)
  ccall( (:cublasStpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Ptr{Cfloat}, Cint), handle, uplo, n, AP, A, lda)
end
function cublasDtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Ptr{Cdouble}, A::Ptr{Cdouble}, lda::Cint)
  ccall( (:cublasDtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}, Cint), handle, uplo, n, AP, A, lda)
end
function cublasCtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Ptr{cuComplex}, A::Ptr{cuComplex}, lda::Cint)
  ccall( (:cublasCtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Ptr{cuComplex}, Cint), handle, uplo, n, AP, A, lda)
end
function cublasZtpttr(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, AP::Ptr{cuDoubleComplex}, A::Ptr{cuDoubleComplex}, lda::Cint)
  ccall( (:cublasZtpttr, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Ptr{cuDoubleComplex}, Cint), handle, uplo, n, AP, A, lda)
end
function cublasStrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Ptr{Cfloat}, lda::Cint, AP::Ptr{Cfloat})
  ccall( (:cublasStrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cfloat}, Cint, Ptr{Cfloat}), handle, uplo, n, A, lda, AP)
end
function cublasDtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Ptr{Cdouble}, lda::Cint, AP::Ptr{Cdouble})
  ccall( (:cublasDtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), handle, uplo, n, A, lda, AP)
end
function cublasCtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Ptr{cuComplex}, lda::Cint, AP::Ptr{cuComplex})
  ccall( (:cublasCtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuComplex}, Cint, Ptr{cuComplex}), handle, uplo, n, A, lda, AP)
end
function cublasZtrttp(handle::cublasHandle_t, uplo::cublasFillMode_t, n::Cint, A::Ptr{cuDoubleComplex}, lda::Cint, AP::Ptr{cuDoubleComplex})
  ccall( (:cublasZtrttp, libcublas), cublasStatus_t, (cublasHandle_t, cublasFillMode_t, Cint, Ptr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex}), handle, uplo, n, A, lda, AP)
end

