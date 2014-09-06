# CUBLAS implementation progress

The following sections list the CUBLAS functions shown on the CUBLAS
documentation page:

http://docs.nvidia.com/cuda/cublas/index.html

## Level 1 (13 functions)

CUBLAS functions:

* [x] amax
* [x] amin
* [x] asum
* [x] axpy
* [x] copy
* [x] dot, dotc, dotu
* [x] nrm2
* [ ] rot (not implemented in julia blas.jl)
* [ ] rotg (not implemented in julia blas.jl)
* [ ] rotm (not implemented in julia blas.jl)
* [ ] rotmg (not implemented in julia blas.jl)
* [x] scal
* [ ] swap (not implemented in julia blas.jl)

## Level 2

CUBLAS functions:

* [ ] gbmv
* [ ] gemv
* [ ] ger
* [ ] sbmv
* [ ] spmv
* [ ] spr
* [ ] spr2
* [ ] symv
* [ ] syr
* [ ] syr2
* [ ] tbmv
* [ ] tbsv
* [ ] tpmv
* [ ] tpsv
* [ ] trmv
* [ ] trsv
* [ ] hemv
* [ ] hbmv
* [ ] hpmv
* [ ] her
* [ ] her2
* [ ] hpr
* [ ] hpr2

## Level 3

CUBLAS functions:

* [ ] gemm
* [ ] gemmBatched
* [ ] symm
* [ ] syrk
* [ ] syr2k
* [ ] syrkx
* [ ] trmm
* [ ] trsm
* [ ] trsmBatched
* [ ] hemm
* [ ] herk
* [ ] her2k
* [ ] herkx

## BLAS-like extensions

* [ ] geam
* [ ] dgmm
* [ ] getrfBatched
* [ ] getriBatched
* [ ] geqrfBatched
* [ ] gelsBatched
* [ ] tpttr
* [ ] trttp
