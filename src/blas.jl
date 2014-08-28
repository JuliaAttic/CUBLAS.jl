# blas.jl
#
# "High level" blas interface to cublas.
# Modeled from julia/src/base/linalg/blas.jl
#

export
# Level 1
    blascopy!,
    scal!,
    scal

function cublascall(s::Symbol)
    return symbol("cublas"*string(s)*"_v2")
end

# Level 1
## copy
for (fname, elty) in ((:Dcopy,:Float64),
                      (:Scopy,:Float32),
                      (:Zcopy,:Complex128),
                      (:Ccopy,:Complex64))
    @eval begin
        # SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
        function blascopy!(n::Integer, DX::CudaArray{$elty}, incx::Integer, DY::CudaArray{$elty}, incy::Integer)
            $(cublascall(fname))(cublashandle[1],n,DX,incx,DY,incy)
            DY
        end
    end
end

## scal
for (fname, elty) in ((:Dscal,:Float64),
                      (:Sscal,:Float32),
                      (:Zscal,:Complex128),
                      (:Cscal,:Complex64))
    @eval begin
        # SUBROUTINE DSCAL(N,DA,DX,INCX)
        function scal!(n::Integer, DA::$elty, DX::CudaArray{$elty}, incx::Integer)
            $(cublascall(fname))(cublashandle[1],n,[DA],DX,incx)
            DX
        end
    end
end
# TODO: implement copy{T}(x::CudaArray{T})
#scal{T}(n::Integer, DA::T, DX::CudaArray{T}, incx::Integer) = scal!(n, DA, copy(DX), incx)
# In case DX is complex, and DA is real, use dscal/sscal to save flops
#for (fname, elty, celty) in ((:Sscal, :Float32, :Complex64),
#                             (:Dscal, :Float64, :Complex128))
#    @eval begin
#        # SUBROUTINE DSCAL(N,DA,DX,INCX)
#        function scal!(n::Integer, DA::$elty, DX::CudaArray{$celty}, incx::Integer)
#            DY = reinterpret($elty,DX,(2*n,))
#            $(cublascall(fname))(cublashandle[1],2*n,[DA],DY,incx)
#            DX
#        end
#    end
#end
