# blas.jl
#
# "High level" blas interface to cublas.
# Modeled from julia/src/base/linalg/blas.jl
#

export
# Level 1
    blascopy!

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
