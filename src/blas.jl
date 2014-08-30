# blas.jl
#
# "High level" blas interface to cublas.
# Modeled from julia/src/base/linalg/blas.jl
#

#function cublascall(s::Symbol)
#    return symbol("cublas"*string(s)*"_v2")
#end

# Level 1
## copy
for (fname, elty) in ((:cublasDcopy_v2,:Float64),
                      (:cublasScopy_v2,:Float32),
                      (:cublasZcopy_v2,:Complex128),
                      (:cublasCcopy_v2,:Complex64))
    @eval begin
        # SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
        function blascopy!(n::Integer,
                           DX::Union(CudaDevicePtr{$elty},CudaArray{$elty}),
                           incx::Integer,
                           DY::Union(CudaDevicePtr{$elty},CudaArray{$elty}),
                           incy::Integer)
              statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                                (cublasHandle_t, Cint, Ptr{$elty}, Cint,
                                 Ptr{$elty}, Cint),
                                cublashandle[1], n, DX, incx, DY, incy))
            DY
        end
    end
end

## scal
for (fname, elty) in ((:cublasDscal_v2,:Float64),
                      (:cublasSscal_v2,:Float32),
                      (:cublasZscal_v2,:Complex128),
                      (:cublasCscal_v2,:Complex64))
    @eval begin
        # SUBROUTINE DSCAL(N,DA,DX,INCX)
        function scal!(n::Integer,
                       DA::$elty,
                       DX::Union(CudaDevicePtr{$elty},CudaArray{$elty}),
                       incx::Integer)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Cint),
                              cublashandle[1], n, [DA], DX, incx))
            DX
        end
    end
end
# TODO: implement copy{T}(x::CudaArray{T})
#scal{T}(n::Integer, DA::T, DX::CudaArray{T}, incx::Integer) = scal!(n, DA, copy(DX), incx)
# In case DX is complex, and DA is real, use dscal/sscal to save flops
for (fname, elty, celty) in ((:cublasSscal_v2, :Float32, :Complex64),
                             (:cublasDscal_v2, :Float64, :Complex128))
    @eval begin
        # SUBROUTINE DSCAL(N,DA,DX,INCX)
        function scal!(n::Integer,
                       DA::$elty,
                       DX::Union(CudaDevicePtr{$celty},CudaArray{$celty}),
                       incx::Integer)
            #DY = reinterpret($elty,DX,(2*n,))
            #$(cublascall(fname))(cublashandle[1],2*n,[DA],DY,incx)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Ptr{$elty}, Ptr{$celty},
                               Cint),
                              cublashandle[1], 2*n, [DA], DX, incx))
            DX
        end
    end
end

## dot
# cublasStatus_t cublasDdot_v2
#   (cublasHandle_t handle,
#    int n,
#    const double *x, int incx,
#    const double *y, int incy,
#    double *result);
for (fname, elty) in ((:cublasDdot_v2,:Float64),
                      (:cublasSdot_v2,:Float32))
    @eval begin
        function dot(n::Integer,
                     DX::Union(CudaDevicePtr{$elty},CudaArray{$elty}),
                     incx::Integer,
                     DY::Union(CudaDevicePtr{$elty},CudaArray{$elty}),
                     incy::Integer)
            result = Array($elty,1)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Ptr{$elty}, Cint,
                               Ptr{$elty}, Cint, Ptr{$elty}),
                              cublashandle[1], n, DX, incx, DY, incy, result))
            return result[1]
        end
    end
end
