using CUBLAS
using CUDArt
using Base.Test

m = 20
n = 35
k = 13

# test blascopy!
function test_blascopy!{T}(A::Array{T})
    @test ndims(A) == 1
    n1 = length(A)
    d_A = CudaArray(A)
    d_B = CudaArray(T,n1)
    blascopy!(n,d_A,1,d_B,1)
    B = to_host(d_B)
    @test A == B
end
test_blascopy!(Float32[1:m])
test_blascopy!(Float64[1:m])
test_blascopy!(Float32[1:m]+im*Float32[1:m])
test_blascopy!(Float64[1:m]+im*Float64[1:m])

# test scal!
function test_scal!{T}(alpha,A::Array{T})
    @test ndims(A) == 1
    n1 = length(A)
    d_A = CudaArray(A)
    scal!(n1,alpha,d_A,1)
    A1 = to_host(d_A)
    @test_approx_eq(alpha*A,A1)
end
test_scal!(2.0f0,Float32[1:m])
test_scal!(2.0,Float64[1:m])
test_scal!(1.0f0+im*1.0f0,Float32[1:m]+im*Float32[1:m])
test_scal!(1.0+im*1.0,Float64[1:m]+im*Float64[1:m])
test_scal!(2.0f0,Float32[1:m]+im*Float32[1:m])
test_scal!(2.0,Float64[1:m]+im*Float64[1:m])
