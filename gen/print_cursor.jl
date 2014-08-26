using Clang.cindex

#indexh = "/usr/local/cuda/include/cublas_v2.h"
indexh = "test.h"
clang_includes = ["/usr/lib/clang/3.4/include", dirname(indexh)]

top = parse_header(indexh; includes=clang_includes, diagnostics=true)

for cu in children(top)
    println(string(typeof(cu))*": "*name(cu))
end
