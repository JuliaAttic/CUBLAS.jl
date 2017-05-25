# CUBLAS.jl

**Build status**: [![](https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUBLAS.jl:%20Julia%200.5%20(x86-64)&badge=Julia%20v0.5)](https://ci.maleadt.net/buildbot/julia/builders/CUBLAS.jl%3A%20Julia%200.5%20%28x86-64%29) [![](https://ci.maleadt.net/buildbot/julia/badge.svg?builder=CUBLAS.jl:%20Julia%200.6%20(x86-64)&badge=Julia%200.6)](https://ci.maleadt.net/buildbot/julia/builders/CUBLAS.jl%3A%20Julia%200.6%20%28x86-64%29)

**Code coverage**: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUBLAS.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUBLAS.jl)

Julia bindings to [NVIDIA's CUDA BLAS](http://docs.nvidia.com/cuda/cublas/#axzz3QuWcFxvY) library.

**Current Status**
* Low level interface to CUBLAS funtionality is implemented.
* High level Julia interface is started. The following are available:
    * `Ax_mul_Bx` for matrices and vectors.
    * `norm`
    * `dot`, `vecdot`
    * `scale`, `scale!`
