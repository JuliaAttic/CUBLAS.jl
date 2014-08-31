import Base.dot
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
    CUBLAS.blascopy!(n,d_A,1,d_B,1)
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
    CUBLAS.scal!(n1,alpha,d_A,1)
    A1 = to_host(d_A)
    @test_approx_eq(alpha*A,A1)
end
test_scal!(2.0f0,Float32[1:m])
test_scal!(2.0,Float64[1:m])
test_scal!(1.0f0+im*1.0f0,Float32[1:m]+im*Float32[1:m])
test_scal!(1.0+im*1.0,Float64[1:m]+im*Float64[1:m])
test_scal!(2.0f0,Float32[1:m]+im*Float32[1:m])
test_scal!(2.0,Float64[1:m]+im*Float64[1:m])

# test dot
function test_dot(A,B)
    @test ndims(A) == 1
    @test ndims(B) == 1
    @test length(A) == length(B)
    n1 = length(A)
    d_A = CudaArray(A)
    d_B = CudaArray(B)
    cuda_dot1 = CUBLAS.dot(n1,d_A,1,d_B,1)
    cuda_dot2 = CUBLAS.dot(d_A,d_B)
    host_dot = dot(A,B)
    @test_approx_eq(cuda_dot1,host_dot)
    @test_approx_eq(cuda_dot2,host_dot)
end
test_dot(Float32[1:m],Float32[1:m])
test_dot(Float64[1:m],Float64[1:m])

# test dotu
function test_dotu(A,B)
    @test ndims(A) == 1
    @test ndims(B) == 1
    @test length(A) == length(B)
    n1 = length(A)
    d_A = CudaArray(A)
    d_B = CudaArray(B)
    cuda_dot1 = CUBLAS.dotu(n1,d_A,1,d_B,1)
    cuda_dot2 = CUBLAS.dotu(d_A,d_B)
    host_dot = A.'*B
    @test_approx_eq(cuda_dot1,host_dot)
    @test_approx_eq(cuda_dot2,host_dot)
end
test_dotu(rand(Complex64,m),rand(Complex64,m))
test_dotu(rand(Complex128,m),rand(Complex128,m))

# test dotc
function test_dotc(A,B)
    @test ndims(A) == 1
    @test ndims(B) == 1
    @test length(A) == length(B)
    n1 = length(A)
    d_A = CudaArray(A)
    d_B = CudaArray(B)
    cuda_dot1 = CUBLAS.dotc(n1,d_A,1,d_B,1)
    cuda_dot2 = CUBLAS.dotc(d_A,d_B)
    host_dot = A'*B
    @test_approx_eq(cuda_dot1,host_dot)
    @test_approx_eq(cuda_dot2,host_dot)
end
test_dotc(rand(Complex64,m),rand(Complex64,m))
test_dotc(rand(Complex128,m),rand(Complex128,m))

# test nrm2
function test_nrm2(A)
    @test ndims(A) == 1
    n1 = length(A)
    d_A = CudaArray(A)
    cuda_nrm2_1 = CUBLAS.nrm2(n1,d_A,1)
    cuda_nrm2_2 = CUBLAS.nrm2(d_A)
    host_nrm2 = norm(A)
    @test_approx_eq(cuda_nrm2_1,host_nrm2)
    @test_approx_eq(cuda_nrm2_2,host_nrm2)
end
test_nrm2(rand(Float32,m))
test_nrm2(rand(Float64,m))
test_nrm2(rand(Complex64,m))
test_nrm2(rand(Complex128,m))

# test asum
function test_asum(A)
    @test ndims(A) == 1
    n1 = length(A)
    d_A = CudaArray(A)
    cuda_asum1 = CUBLAS.asum(n1,d_A,1)
    cuda_asum2 = CUBLAS.asum(d_A)
    host_asum = sum(abs(real(A)) + abs(imag(A)))
    @test_approx_eq(cuda_asum1,host_asum)
    @test_approx_eq(cuda_asum2,host_asum)
end
test_asum(Float32[1:m])
test_asum(Float64[1:m])
test_asum(rand(Complex64,m))
test_asum(rand(Complex128,m))

# test axpy!
function test_axpy!(alpha,A,B)
    @test length(A) == length(B)
    n1 = length(A)
    d_A = CudaArray(A)
    d_B1 = CudaArray(B)
    CUBLAS.axpy!(n1,alpha,d_A,1,d_B1,1)
    B1 = to_host(d_B1)
    host_axpy = alpha*A + B
    @test_approx_eq(host_axpy,B1)
end
test_axpy!(2.0f0,rand(Float32,m),rand(Float32,m))
test_axpy!(2.0,rand(Float64,m),rand(Float64,m))
test_axpy!(2.0f0+im*2.0f0,rand(Complex64,m),rand(Complex64,m))
test_axpy!(2.0+im*2.0,rand(Complex128,m),rand(Complex128,m))
