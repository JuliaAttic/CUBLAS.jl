# blas.jl
#
# "High level" blas interface to cublas.
# Modeled from julia/src/base/linalg/blas.jl
#

# Level 1
## copy
for (fname, elty) in ((:cublasDcopy_v2,:Float64),
                      (:cublasScopy_v2,:Float32),
                      (:cublasZcopy_v2,:Complex128),
                      (:cublasCcopy_v2,:Complex64))
    @eval begin
        # SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
        function blascopy!(n::Integer,
                           DX::Union(CudaPtr{$elty},CudaArray{$elty}),
                           incx::Integer,
                           DY::Union(CudaPtr{$elty},CudaArray{$elty}),
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
                       DX::Union(CudaPtr{$elty},CudaArray{$elty}),
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
                       DX::Union(CudaPtr{$celty},CudaArray{$celty}),
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

## dot, dotc, dotu
# cublasStatus_t cublasDdot_v2
#   (cublasHandle_t handle,
#    int n,
#    const double *x, int incx,
#    const double *y, int incy,
#    double *result);
for (jname, fname, elty) in ((:dot,:cublasDdot_v2,:Float64),
                             (:dot,:cublasSdot_v2,:Float32),
                             (:dotc,:cublasZdotc_v2,:Complex128),
                             (:dotc,:cublasCdotc_v2,:Complex64),
                             (:dotu,:cublasZdotu_v2,:Complex128),
                             (:dotu,:cublasCdotu_v2,:Complex64))
    @eval begin
        function $jname(n::Integer,
                        DX::Union(CudaPtr{$elty},CudaArray{$elty}),
                        incx::Integer,
                        DY::Union(CudaPtr{$elty},CudaArray{$elty}),
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
# TODO: inspect blas.jl in julia to correct types here
function dot{T<:Union(Float32,Float64)}(DX::CudaArray{T}, DY::CudaArray{T})
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dot(n, DX, 1, DY, 1)
end
function dotc{T<:Union(Complex64,Complex128)}(DX::CudaArray{T}, DY::CudaArray{T})
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotc(n, DX, 1, DY, 1)
end
function dotu{T<:Union(Complex64,Complex128)}(DX::CudaArray{T}, DY::CudaArray{T})
    n = length(DX)
    n==length(DY) || throw(DimensionMismatch("dot product arguments have lengths $(length(DX)) and $(length(DY))"))
    dotu(n, DX, 1, DY, 1)
end

## nrm2
for (fname, elty, ret_type) in ((:cublasDnrm2_v2,:Float64,:Float64),
                                (:cublasSnrm2_v2,:Float32,:Float32),
                                (:cublasDznrm2_v2,:Complex128,:Float64),
                                (:cublasScnrm2_v2,:Complex64,:Float32))
    @eval begin
        # SUBROUTINE DNRM2(N,X,INCX)
        function nrm2(n::Integer,
                      X::Union(CudaPtr{$elty},CudaArray{$elty}),
                      incx::Integer)
            result = Array($ret_type,1)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Ptr{$elty}, Cint,
                               Ptr{$ret_type}),
                              cublashandle[1], n, X, incx, result))
            return result[1]
        end
    end
end
# TODO: consider CudaVector and CudaStridedVector
#nrm2(x::StridedVector) = nrm2(length(x), x, stride(x,1))
nrm2(x::CudaArray) = nrm2(length(x), x, 1)

## asum
for (fname, elty, ret_type) in ((:cublasDasum_v2,:Float64,:Float64),
                                (:cublasSasum_v2,:Float32,:Float32),
                                (:cublasDzasum_v2,:Complex128,:Float64),
                                (:cublasScasum_v2,:Complex64,:Float32))
    @eval begin
        # SUBROUTINE ASUM(N, X, INCX)
        function asum(n::Integer,
                      X::Union(CudaPtr{$elty},CudaArray{$elty}),
                      incx::Integer)
            result = Array($ret_type,1)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Ptr{$elty}, Cint,
                               Ptr{$ret_type}),
                              cublashandle[1], n, X, incx, result))
            return result[1]
        end
    end
end
#asum(x::StridedVector) = asum(length(x), x, stride(x,1))
asum(x::CudaArray) = asum(length(x), pointer(x), 1)

## axpy
for (fname, elty) in ((:cublasDaxpy_v2,:Float64),
                      (:cublasSaxpy_v2,:Float32),
                      (:cublasZaxpy_v2,:Complex128),
                      (:cublasCaxpy_v2,:Complex64))
    @eval begin
        # SUBROUTINE DAXPY(N,DA,DX,INCX,DY,INCY)
        # DY <- DA*DX + DY
        # cublasStatus_t cublasSaxpy_v2(
        #   cublasHandle_t handle,
        #   int n,
        #   const float *alpha, /* host or device pointer */
        #   const float *x,
        #   int incx,
        #   float *y,
        #   int incy);
        function axpy!(n::Integer,
                       alpha::($elty),
                       dx::Union(CudaPtr{$elty},CudaArray{$elty}),
                       incx::Integer,
                       dy::Union(CudaPtr{$elty},CudaArray{$elty}),
                       incy::Integer)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Ptr{$elty}, Ptr{$elty},
                               Cint, Ptr{$elty},
                               Cint),
                              cublashandle[1], n, &alpha, dx, incx, dy, incy))
            dy
        end
    end
end

# TODO: write test for this function
function axpy!{T<:CublasFloat,Ta<:Number}(alpha::Ta,
                                          x::CudaArray{T},
                                          y::CudaArray{T})
    length(x)==length(y) || throw(DimensionMismatch(""))
    axpy!(length(x), convert(T,alpha), x, 1, y, 1)
end

# TODO: implement pointer arithmetic for CudaPtr
#function axpy!{T<:CublasFloat,Ta<:Number,Ti<:Integer}(alpha::Ta,
#                                                      x::CudaArray{T},
#                                                      rx::Union(UnitRange{Ti},Range{Ti}),
#                                                      y::CudaArray{T},
#                                                      ry::Union(UnitRange{Ti},Range{Ti}))
#    length(rx)==length(ry) || throw(DimensionMismatch(""))
#    if minimum(rx) < 1 || maximum(rx) > length(x) || minimum(ry) < 1 || maximum(ry) > length(y)
#        throw(BoundsError())
#    end
#    axpy!(length(rx), convert(T, alpha), pointer(x)+(first(rx)-1)*sizeof(T), step(rx), pointer(y)+(first(ry)-1)*sizeof(T), step(ry))
#    y
#end
