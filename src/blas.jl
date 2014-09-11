# blas.jl
#
# "High level" blas interface to cublas.
# Modeled from julia/src/base/linalg/blas.jl
#
# Author: Nick Henderson <nwh@stanford.edu>
# Created: 2014-08-26
# License: MIT
#

# Utility functions

# convert BlasChar {N,T,C} to cublasOperation_t
function cublasop(trans::BlasChar)
    if trans == 'N'
        return CUBLAS_OP_N
    end
    if trans == 'T'
        return CUBLAS_OP_T
    end
    if trans == 'C'
        return CUBLAS_OP_C
    end
    throw("unknown cublas operation.")
end

# convert BlasChar {U,L} to cublasFillMode_t
function cublasfill(uplo::BlasChar)
    if uplo == 'U'
        return CUBLAS_FILL_MODE_UPPER
    end
    if uplo == 'L'
        return CUBLAS_FILL_MODE_LOWER
    end
    throw("unknown cublas fill mode")
end

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

function axpy!{T<:CublasFloat,Ta<:Number}(alpha::Ta,
                                          x::CudaArray{T},
                                          y::CudaArray{T})
    length(x)==length(y) || throw(DimensionMismatch(""))
    axpy!(length(x), convert(T,alpha), x, 1, y, 1)
end

function axpy!{T<:CublasFloat,Ta<:Number,Ti<:Integer}(alpha::Ta,
                                                      x::CudaArray{T},
                                                      rx::Union(UnitRange{Ti},Range{Ti}),
                                                      y::CudaArray{T},
                                                      ry::Union(UnitRange{Ti},Range{Ti}))
    length(rx)==length(ry) || throw(DimensionMismatch(""))
    if minimum(rx) < 1 || maximum(rx) > length(x) || minimum(ry) < 1 || maximum(ry) > length(y)
        throw(BoundsError())
    end
    axpy!(length(rx), convert(T, alpha), pointer(x)+(first(rx)-1)*sizeof(T),
          step(rx), pointer(y)+(first(ry)-1)*sizeof(T), step(ry))
    y
end

## iamax
# TODO: fix iamax in julia base
for (fname, elty) in ((:cublasIdamax_v2,:Float64),
                      (:cublasIsamax_v2,:Float32),
                      (:cublasIzamax_v2,:Complex128),
                      (:cublasIcamax_v2,:Complex64))
    @eval begin
        function iamax(n::Integer,
                       dx::Union(CudaPtr{$elty}, CudaArray{$elty}),
                       incx::Integer)
            result = Array(Cint,1)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Ptr{$elty}, Cint,
                               Ptr{Cint}),
                              cublashandle[1], n, dx, incx, result))
            return result[1]
        end
    end
end
iamax(dx::CudaArray) = iamax(length(dx), dx, 1)

## iamin
# iamin is not in standard blas is a CUBLAS extension
for (fname, elty) in ((:cublasIdamin_v2,:Float64),
                      (:cublasIsamin_v2,:Float32),
                      (:cublasIzamin_v2,:Complex128),
                      (:cublasIcamin_v2,:Complex64))
    @eval begin
        function iamin(n::Integer,
                       dx::Union(CudaPtr{$elty}, CudaArray{$elty}),
                       incx::Integer)
            result = Array(Cint,1)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, Cint, Ptr{$elty}, Cint,
                               Ptr{Cint}),
                              cublashandle[1], n, dx, incx, result))
            return result[1]
        end
    end
end
iamin(dx::CudaArray) = iamin(length(dx), dx, 1)

# Level 2
## mv
### gemv
for (fname, elty) in ((:cublasDgemv_v2,:Float64),
                      (:cublasSgemv_v2,:Float32),
                      (:cublasZgemv_v2,:Complex128),
                      (:cublasCgemv_v2,:Complex64))
    @eval begin
        # cublasStatus_t cublasDgemv(
        #   cublasHandle_t handle, cublasOperation_t trans,
        #   int m, int n,
        #   const double *alpha,
        #   const double *A, int lda,
        #   const double *x, int incx,
        #   const double *beta,
        #   double *y, int incy)
        function gemv!(trans::BlasChar,
                       alpha::($elty),
                       A::CudaMatrix{$elty},
                       X::CudaVector{$elty},
                       beta::($elty),
                       Y::CudaVector{$elty})
            # handle trans
            cutrans = cublasop(trans)
            m,n = size(A)
            # check dimensions
            length(X) == (trans == 'N' ? n : m) && length(Y) == (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            # compute increments
            lda = max(1,stride(A,2))
            incx = stride(X,1)
            incy = stride(Y,1)
            statuscheck(ccall(($(string(fname)), libcublas), cublasStatus_t,
                              (cublasHandle_t, cublasOperation_t, Cint, Cint,
                              Ptr{$elty}, Ptr{$elty}, Cint, Ptr{$elty},
                              Cint, Ptr{$elty}, Ptr{$elty}, Cint), cublashandle[1],
                              cutrans, m, n, [alpha], A, lda, X, incx, [beta], Y,
                              incy))
            Y
        end
        function gemv(trans::BlasChar, alpha::($elty), A::CudaMatrix{$elty}, X::CudaVector{$elty})
            gemv!(trans, alpha, A, X, zero($elty), similar(X, $elty, size(A, (trans == 'N' ? 1 : 2))))
        end
        function gemv(trans::BlasChar, A::CudaMatrix{$elty}, X::CudaVector{$elty})
            gemv!(trans, one($elty), A, X, zero($elty), similar(X, $elty, size(A, (trans == 'N' ? 1 : 2))))
        end
    end
end

### (GB) general banded matrix-vector multiplication
for (fname, elty) in ((:cublasDgbmv_v2,:Float64),
                      (:cublasSgbmv_v2,:Float32),
                      (:cublasZgbmv_v2,:Complex128),
                      (:cublasCgbmv_v2,:Complex64))
    @eval begin
        # cublasStatus_t cublasDgbmv(
        #   cublasHandle_t handle, cublasOperation_t trans,
        #   int m, int n, int kl, int ku,
        #   const double *alpha, const double *A, int lda,
        #   const double *x, int incx,
        #   const double *beta, double *y, int incy)
        function gbmv!(trans::BlasChar,
                       m::Integer,
                       kl::Integer,
                       ku::Integer,
                       alpha::($elty),
                       A::CudaMatrix{$elty},
                       x::CudaVector{$elty},
                       beta::($elty),
                       y::CudaVector{$elty})
            # handle trans
            cutrans = cublasop(trans)
            n = size(A,2)
            # check dimensions
            length(x) == (trans == 'N' ? n : m) && length(y) == (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            # compute increments
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            statuscheck(ccall(($(string(fname)),libcublas), cublasStatus_t,
                              (cublasHandle_t, cublasOperation_t, Cint, Cint,
                               Cint, Cint, Ptr{$elty}, Ptr{$elty}, Cint,
                               Ptr{$elty}, Cint, Ptr{$elty}, Ptr{$elty},
                               Cint), cublashandle[1], cutrans, m, n, kl, ku, [alpha], A,
                              lda, x, incx, [beta], y, incy))
            y
        end
        function gbmv(trans::BlasChar,
                      m::Integer,
                      kl::Integer,
                      ku::Integer,
                      alpha::($elty),
                      A::CudaMatrix{$elty},
                      x::CudaVector{$elty})
            # TODO: fix gbmv bug in julia
            n = size(A,2)
            leny = trans == 'N' ? m : n
            gbmv!(trans, m, kl, ku, alpha, A, x, zero($elty), similar(x, $elty, leny))
        end
        function gbmv(trans::BlasChar,
                      m::Integer,
                      kl::Integer,
                      ku::Integer,
                      A::CudaMatrix{$elty},
                      x::CudaVector{$elty})
            gbmv(trans, m, kl, ku, one($elty), A, x)
        end
    end
end

### symv
for (fname, elty) in ((:cublasDsymv_v2,:Float64),
                      (:cublasSsymv_v2,:Float32),
                      (:cublasZsymv_v2,:Complex128),
                      (:cublasCsymv_v2,:Complex64))
    # Note that the complex symv are not BLAS but auiliary functions in LAPACK
    @eval begin
        # cublasStatus_t cublasDsymv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   int n, const double *alpha, const double *A, int lda,
        #   const double *x, int incx,
        #   const double *beta, double *y, int incy)
        function symv!(uplo::BlasChar,
                       alpha::($elty),
                       A::CudaMatrix{$elty},
                       x::CudaVector{$elty},
                       beta::($elty),
                       y::CudaVector{$elty})
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            statuscheck(ccall(($(string(fname)),libcublas), cublasStatus_t,
                              (cublasHandle_t, cublasFillMode_t,
                              Cint,Ptr{$elty}, Ptr{$elty}, Cint,
                              Ptr{$elty}, Cint, Ptr{$elty},
                              Ptr{$elty},Cint),
                              cublashandle[1], cuuplo, n, [alpha],
                              A, lda, x, incx, [beta], y, incy))
            y
        end
        function symv(uplo::BlasChar, alpha::($elty), A::CudaMatrix{$elty}, x::CudaVector{$elty})
                symv!(uplo, alpha, A, x, zero($elty), similar(x))
        end
        function symv(uplo::BlasChar, A::CudaMatrix{$elty}, x::CudaVector{$elty})
            symv(uplo, one($elty), A, x)
        end
    end
end

### hemv
# TODO: fix chemv_ function call bug in julia
for (fname, elty) in ((:cublasZhemv_v2,:Complex128),
                      (:cublasChemv_v2,:Complex64))
    @eval begin
        # cublasStatus_t cublasChemv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   int n, const cuComplex *alpha, const cuComplex *A, int lda,
        #   const cuComplex *x, int incx,
        #   const cuComplex *beta, cuComplex *y, int incy)
        function hemv!(uplo::BlasChar,
                       alpha::$elty,
                       A::CudaMatrix{$elty},
                       x::CudaVector{$elty},
                       beta::$elty,
                       y::CudaVector{$elty})
            # TODO: fix dimension check bug in julia
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            statuscheck(ccall(($(string(fname)),libcublas), cublasStatus_t,
                              (cublasHandle_t, cublasFillMode_t,
                              Cint,Ptr{$elty}, Ptr{$elty}, Cint,
                              Ptr{$elty}, Cint, Ptr{$elty},
                              Ptr{$elty},Cint),
                              cublashandle[1], cuuplo, n, [alpha],
                              A, lda, x, incx, [beta], y, incy))
            y
        end
        function hemv(uplo::BlasChar, alpha::($elty), A::CudaMatrix{$elty},
                      x::CudaVector{$elty})
            hemv!(uplo, alpha, A, x, zero($elty), similar(x))
        end
        function hemv(uplo::BlasChar, A::CudaMatrix{$elty},
                      x::CudaVector{$elty})
            hemv(uplo, one($elty), A, x)
        end
    end
end
