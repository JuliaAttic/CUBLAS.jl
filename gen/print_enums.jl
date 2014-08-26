using Clang.cindex

indexh = "/usr/local/cuda/include/cublas_v2.h"
clang_includes = ["/usr/lib/clang/3.4/include", dirname(indexh)]

top = parse_header(indexh; includes=clang_includes, diagnostics=true)

function print_enums(enumdef::EnumDecl)
    for enum in children(enumdef)
        println("  const ", name(enum), " = ", value(enum))
    end
end

for cursor in children(top)
    # Skip cursors not in target file
    #if (basename(cu_file(cursor)) != basename(indexh)) continue end

    got_enum = false
    if isa(cursor, EnumDecl)
        #if (name(cursor) == "") continue end
        println("# Enum: ", name(cursor))
        print_enums(cursor)
        got_enum = true
    elseif isa(cursor, TypedefDecl)
        td_children = children(cursor)
        td_children.size == 0 && continue

        td_cursor = td_children[1]
        if isa(td_cursor, EnumDecl)
            println("# Typedef Enum: ", name(td_cursor))
            print_enums(td_cursor)
            got_enum = true
        end
    end
end
