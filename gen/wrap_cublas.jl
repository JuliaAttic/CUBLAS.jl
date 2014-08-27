# wrap_cublas.jl
#
# Script to generate CUBLAS wraper code. Generated code in libcublas_types.jl
# file needs to be modified before including in the CUBLAS module. Code is
# generated here and manually moved to the ../src directory.
#
# This script works with Clang.jl v0.0.1.  To install and use on Fedora 20:
#   julia> ENV["LLVM_CONFIG"]="/usr/bin/llvm-config"
#   julia> Pkg.add("Clang")
#   julia> Pkg.pin("Clang",v"0.0.1")
#   julia> Pkg.build("Clang")
#
# Note: this script needs to be changed for Clang.jl v0.0.2
#
# Author: Nick Henderson <nwh@stanford.edu>
# Created: 2014-08-26
# License: MIT
#

using Clang, Clang.cindex
import Clang.wrap_c: repr_jl, rep_type, rep_args, name_safe

# main cublas header
headers = ["/usr/local/cuda/include/cublas_v2.h"]

# may need to be modified to host system
includes = ["/usr/include",
            "/usr/lib/gcc/x86_64-redhat-linux/4.8.3"]

# skip cursors with empty names, this avoids anonymous enums
function check_anon(cursorname, cursor)
    if isempty(cursorname)
        return false
    end
    return true
end

# overload wrap_c.wrap to ignore CUBLAS macro definitions
function wrap_c.wrap(strm::IO, md::cindex.MacroDefinition)
    warn("macro definition")
end

# Customize the wrap function for functions. This was copied
# from Clang/src/wrap_c.jl, with the following customizations:
#   - error-check functions that return a cublasStatus_t
#   - omit types from function prototypes
# This function was copied from CUDArt.jl/gen/wrap_cuda.jl
skipcheck = []
function wrap_c.wrap(buf::IO, funcdecl::FunctionDecl, libname::ASCIIString)
    function print_args(buf::IO, cursors, types)
        i = 1
        for c in cursors
            print(buf, name_safe(c))
            (i < length(cursors)) && print(buf, ", ")
            i += 1
        end
    end

    cu_spelling = spelling(funcdecl)

    funcname = spelling(funcdecl)
    arg_types = cindex.function_args(funcdecl)
    args = [x for x in search(funcdecl, ParmDecl)]
    arg_list = tuple( [repr_jl(x) for x in arg_types]... )
    ret_type = repr_jl(return_type(funcdecl))

    print(buf, "function ")
    fname = spelling(funcdecl)
    print(buf, fname)
    print(buf, "(")
    print_args(buf, args, [myrepr_jl(x) for x in arg_types])
    println(buf, ")")
    print(buf, "  ")
    checkerr = ret_type == "cublasStatus_t" && !in(fname,skipcheck)
    checkerr && print(buf, "statuscheck(")
    print(buf, "ccall( (:", funcname, ", ", libname, "), ")
    print(buf, rep_type(ret_type))
    print(buf, ", ")
    print(buf, rep_args(arg_list), ", ")
    for (i,arg) in enumerate(args)
        print(buf, name_safe(arg))
        (i < length(args)) && print(buf, ", ")
    end
    checkerr && print(buf, ")")
    println(buf, ")")
    println(buf, "end")
end

function myrepr_jl(x)
    str = repr_jl(x)
    return str
    #return (str == "Ptr{Cint}") ? "Array{Cint}" : str
end

# initialize wrap context
context = wrap_c.init(output_file="libcublas.jl",
                      common_file="libcublas_types.jl",
                      header_library=x->"libcublas",
                      clang_includes=includes,
                      cursor_wrapped=check_anon,
                      header_wrapped=(x,y)->(contains(y,"cublas")))

# generate the wrapper
wrap_c.wrap_c_headers(context,headers)
