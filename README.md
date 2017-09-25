### Note: This package is being phased out.

The same functionality is available with [CuArrays](https://github.com/JuliaGPU/CuArrays.jl).

# CUBLAS.jl

**Build status**: [![][buildbot-julia05-img]][buildbot-julia05-url] [![][buildbot-julia06-img]][buildbot-julia06-url]

**Code coverage**: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUBLAS.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUBLAS.jl)

[buildbot-julia05-img]: http://ci.maleadt.net/shields/build.php?builder=CUBLAS-julia05-x86-64bit&name=julia%200.5
[buildbot-julia05-url]: http://ci.maleadt.net/shields/url.php?builder=CUBLAS-julia05-x86-64bit
[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CUBLAS-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CUBLAS-julia06-x86-64bit

Julia bindings to [NVIDIA's CUDA BLAS](http://docs.nvidia.com/cuda/cublas/#axzz3QuWcFxvY) library.

**Current Status**
* Low level interface to CUBLAS funtionality is implemented.
* High level Julia interface is started. The following are available:
    * `Ax_mul_Bx` for matrices and vectors.
    * `norm`
    * `dot`, `vecdot`
    * `scale`, `scale!`
