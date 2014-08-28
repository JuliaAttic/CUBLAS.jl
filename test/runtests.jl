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
