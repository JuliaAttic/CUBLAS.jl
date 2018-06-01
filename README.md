### Note: This package is being phased out.

The same functionality is available with [CuArrays](https://github.com/JuliaGPU/CuArrays.jl).

# CUBLAS.jl

**Code coverage**: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUBLAS.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUBLAS.jl)

Julia bindings to [NVIDIA's CUDA BLAS](http://docs.nvidia.com/cuda/cublas/#axzz3QuWcFxvY) library.

**Current Status**
* Low level interface to CUBLAS funtionality is implemented.
* High level Julia interface is started. The following are available:
    * `Ax_mul_Bx` for matrices and vectors.
    * `norm`
    * `dot`, `vecdot`
    * `scale`, `scale!`
