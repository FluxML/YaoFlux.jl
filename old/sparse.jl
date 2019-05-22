using Zygote, SparseArrays

Zygote.@adjoint Base.:(-)(a, b) = a-b, Δ -> (Δ, -Δ)


_, back = Zygote._forward(sprand(2, 2, 0.1), sprand(2, 2, 0.2)) do x, y
    x - y
end
