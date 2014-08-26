using Clang.wrap_c
using Clang.cindex

includes = ["/usr/include",
            "/usr/lib/gcc/x86_64-redhat-linux/4.8.3"]
headers = ["/usr/local/cuda/include/cublas_v2.h"]

function check_anon(cursorname, cursor)
    # ignore anonymous cursors
    if isempty(cursorname)
        return false
    end
    #if typeof(cursor) == cindex.MacroDefinition
    #    return false
    #end
    return true
end

function wrap_c.wrap(strm::IO, md::cindex.MacroDefinition)
    warn("macro definition")
end

#check_anon(cn,cu::cindex.MacroDefinition) = false

# initialize wrap context
context = wrap_c.init(output_file="libcublas.jl",
                      common_file="libcublas_h.jl",
                      header_library=x->"libcublas",
                      clang_includes=includes,
                      cursor_wrapped=check_anon,
                      header_wrapped=(x,y)->(contains(y,"cublas")))

#function wrap(context::WrapContext, expr_buf::OrderedDict, md::cindex.MacroDefinition)
#    warn("macro definition")
#end

# wrap structs
context.options = wrap_c.InternalOptions(true)
# generate the wrapper
wrap_c.wrap_c_headers(context,headers)
