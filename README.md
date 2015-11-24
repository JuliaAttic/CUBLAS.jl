# CUBLAS.jl

[![Build Status](https://travis-ci.org/JuliaGPU/CUBLAS.jl.svg?branch=master)](https://travis-ci.org/JuliaGPU/CUBLAS.jl)

Julia bindings to [NVIDIA's CUDA BLAS](http://docs.nvidia.com/cuda/cublas/#axzz3QuWcFxvY) library.

**Current Status**
* Low level interface to CUBLAS funtionality is implemented.
* High level Julia interface is started. The following are available:
    * `Ax_mul_Bx` for matrices and vectors.
    * `norm`
    * `dot`, `vecdot`
    * `scale`, `scale!`
