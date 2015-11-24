import Base.Operators.(*)

import Base: scale!, scale, norm, vecdot

import Base: A_mul_B!, At_mul_B,  Ac_mul_B,  A_mul_Bc,  At_mul_Bt,  Ac_mul_Bc,  At_mul_Bt,
                       At_mul_B!, Ac_mul_B!, A_mul_Bc!, At_mul_Bt!, Ac_mul_Bc!, At_mul_Bt!

cublas_size(t::Char, M::CudaVecOrMat) = (size(M, t=='N' ? 1:2), size(M, t=='N' ? 2:1))

###########
#
# BLAS 1
#
###########

#######
# SCAL
#######
scale!{T<:CublasFloat}(x::CudaArray{T}, k::Number) = CUBLAS.scal!(length(x), k, x, 1)
scale{T<:CublasFloat}(x::CudaArray{T}, k::Number) = CUBLAS.scal!(length(x), k, copy(x), 1)

#######
# DOT
#######
function dot{T <: CublasFloat, TI<:Integer}(x::CudaVector{T}, rx::Union{UnitRange{TI},Range{TI}}, y::CudaVector{T}, ry::Union{UnitRange{TI},Range{TI}})
    if length(rx) != length(ry)
        throw(DimensionMismatch("length of rx, $(length(rx)), does not equal length of ry, $(length(ry))"))
    end
    if minimum(rx) < 1 || maximum(rx) > length(x)
        throw(BoundsError(x, rx))
    end
    if minimum(ry) < 1 || maximum(ry) > length(y)
        throw(BoundsError(y, ry))
    end
    dot(length(rx), pointer(x)+(first(rx)-1)*sizeof(T), step(rx), pointer(y)+(first(ry)-1)*sizeof(T), step(ry))
end

At_mul_B{T<:CublasReal}(x::CudaVector{T}, y::CudaVector{T}) = [CUBLAS.dot(x, y)]
At_mul_B{T<:CublasComplex}(x::CudaVector{T}, y::CudaVector{T}) = [CUBLAS.dotu(x, y)]
Ac_mul_B{T<:CublasComplex}(x::CudaVector{T}, y::CudaVector{T}) = [CUBLAS.dotc(x, y)]

vecdot{T<:CublasReal}(x::CudaVector{T}, y::CudaVector{T}) = dot(x, y)
vecdot{T<:CublasComplex}(x::CudaVector{T}, y::CudaVector{T}) = dotc(x, y)

#######
# NRM2
#######
norm(x::CudaArray) = nrm2(x)


############
#
# BLAS 2
#
############


#########
# GEMV
##########
function gemv_wrapper!{T<:CublasFloat}(y::CudaVector{T}, tA::Char, A::CudaMatrix{T}, x::CudaVector{T},
                                       alpha = one(T), beta = zero(T))
    mA, nA = cublas_size(tA, A)
    if nA != length(x)
        throw(DimensionMismatch("second dimension of A, $nA, does not match length of x, $(length(x))"))
    end
    if mA != length(y)
        throw(DimensionMismatch("first dimension of A, $mA, does not match length of y, $(length(y))"))
    end
    if mA == 0
        return y
    end
    if nA == 0
        return scale!(y, 0)
    end
    gemv!(tA, alpha, A, x, beta, y)
end

A_mul_B!{T<:CublasFloat}(y::CudaVector{T}, A::CudaMatrix{T}, x::CudaVector{T}) = gemv_wrapper!(y, 'N', A,  x)
At_mul_B!{T<:CublasFloat}(y::CudaVector{T}, A::CudaMatrix{T}, x::CudaVector{T}) = gemv_wrapper!(y, 'T', A, x)
Ac_mul_B!{T<:CublasFloat}(y::CudaVector{T}, A::CudaMatrix{T}, x::CudaVector{T}) = gemv_wrapper!(y, 'T', A, x)
Ac_mul_B!{T<:CublasComplex}(y::CudaVector{T}, A::CudaMatrix{T}, x::CudaVector{T}) = gemv_wrapper!(y, 'C', A, x)

function (*){T<:CublasFloat}(A::CudaMatrix{T}, x::CudaVector{T})
    A_mul_B!(similar(x, T, size(A,1)), A, x)
end

function At_mul_B{T<:CublasFloat}(A::CudaMatrix{T}, x::CudaVector{T})
    At_mul_B!(similar(x, T, size(A,2)), A, x)
end

function Ac_mul_B{T<:CublasFloat}(A::CudaMatrix{T}, x::CudaVector{T})
    Ac_mul_B!(similar(x, T, size(A,2)), A, x)
end

############
#
# BLAS 3
#
############


########
# GEMM
########
function gemm_wrapper!{T <: CublasFloat}(C::CudaVecOrMat{T}, tA::Char, tB::Char,
                                     A::CudaVecOrMat{T},
                                     B::CudaVecOrMat{T},
                                     alpha = one(T),
                                     beta = zero(T))
    mA, nA = cublas_size(tA, A)
    mB, nB = cublas_size(tB, B)

    if nA != mB
        throw(DimensionMismatch("A has dimensions ($mA,$nA) but B has dimensions ($mB,$nB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    if mA == 0 || nA == 0 || nB == 0
        if size(C) != (mA, nB)
            throw(DimensionMismatch("C has dimensions $(size(C)), should have ($mA,$nB)"))
        end
        return scale!(C, 0)
    end

    gemm!(tA, tB, alpha, A, B, beta, C)
end

# Mutating
A_mul_B!{T <: CublasFloat}(C::CudaMatrix{T}, A::CudaMatrix{T}, B::CudaMatrix{T}) = gemm_wrapper!(C, 'N', 'N', A, B)
At_mul_B!(C::CudaMatrix, A::CudaMatrix, B::CudaMatrix) = gemm_wrapper!(C, 'T', 'N', A, B)
At_mul_Bt!(C::CudaMatrix, A::CudaMatrix, B::CudaMatrix) = gemm_wrapper!(C, 'T', 'T', A, B)
Ac_mul_B!{T<:CublasReal}(C::CudaMatrix{T}, A::CudaMatrix{T}, B::CudaMatrix{T}) = At_mul_B!(C, A, B)
Ac_mul_B!(C::CudaMatrix, A::CudaMatrix, B::CudaMatrix) = gemm_wrapper!(C, 'C', 'N', A, B)

function A_mul_B!{T}(C::CudaMatrix{T}, A::CudaVecOrMat{T}, B::CudaVecOrMat{T})
    gemm_wrapper!(C, 'N', 'N', A, B)
end

# Non mutating

# A_mul_Bx
function (*){T <: CublasFloat}(A::CudaMatrix{T}, B::CudaMatrix{T})
    A_mul_B!(similar(B, T,(size(A,1), size(B,2))), A, B)
end

function A_mul_Bt{T}(A::CudaMatrix{T}, B::CudaMatrix{T})
    A_mul_Bt!(similar(B, T, (size(A,1), size(B,1))), A, B)
end

function A_mul_Bc{T}(A::CudaMatrix{T}, B::CudaMatrix{T})
    A_mul_Bc!(similar(B, T,(size(A,1),size(B,1))),A, B)
end

# At_mul_Bx
function At_mul_B{T}(A::CudaMatrix{T}, B::CudaMatrix{T})
    At_mul_B!(similar(B, T, (size(A,2), size(B,2))), A, B)
end

function At_mul_Bt{T}(A::CudaMatrix{T}, B::CudaMatrix{T})
    At_mul_Bt!(similar(B, T, (size(A,2), size(B,1))), A, B)
end

# Ac_mul_Bx
function Ac_mul_B{T}(A::CudaMatrix{T}, B::CudaMatrix{T})
    Ac_mul_B!(similar(B, T, (size(A,2), size(B,2))), A, B)
end

function Ac_mul_Bt{T,S}(A::CudaMatrix{T}, B::CudaMatrix{S})
    Ac_mul_Bt(similar(B, T, (size(A,2), size(B,1))), A, B)
end

function Ac_mul_Bc{T,S}(A::CudaMatrix{T}, B::CudaMatrix{S})
    Ac_mul_Bc!(similar(B, T, (size(A,2), size(B,1))), A, B)
end


