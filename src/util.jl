function band(A::AbstractMatrix,kl,ku)
    m, n = size(A)
    AB = zeros(eltype(A),kl+ku+1,n)
    for j = 1:n
        for i = max(1,j-ku):min(m,j+kl)
            AB[ku+1-j+i,j] = A[i,j]
        end
    end
    return AB
end

function unband(AB::AbstractMatrix,m,kl,ku)
    bm, n = size(AB)
    A = zeros(eltype(AB),m,n)
    for j = 1:n
        for i = max(1,j-ku):min(m,j+kl)
            A[i,j] = AB[ku+1-j+i,j]
        end
    end
    return A
end
