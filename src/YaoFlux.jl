module YaoFlux

using Zygote, LuxurySparse, Yao, YaoBase, SparseArrays, BitBasis
import Zygote: Context

Zygote.@adjoint (::Type{T})(perms, vals) where T <: PermMatrix = T(perms, vals), Δ -> nothing
Zygote.@adjoint (::Type{T})() where T <: IMatrix = T(), Δ -> nothing
Zygote.@adjoint Base.:(*)(A::Number, B::PermMatrix) = A * B, Δ->(sum(Δ .* B), A * Δ)
Zygote.@adjoint Base.:(*)(A::PermMatrix, B::Number) = A * B, Δ->(A * Δ, sum(Δ .* B))

Zygote.@adjoint SparseArrays.SparseMatrixCSC(A::PermMatrix) = SparseMatrixCSC(A), Δ->(Δ, )

Zygote.@adjoint BitBasis.onehot(::Type{T}, nbits::Int, x::Integer, nbatch::Int) where T = onehot(T, nbits, x, nbatch), Δ -> nothing

# upstreams
Zygote.@adjoint Base.:(-)(a, b) = a-b, Δ -> (Δ, -Δ)
Zygote.@adjoint Base.:(+)(a, b) = a+b, Δ -> (Δ, Δ)

# require mutate
Zygote.@adjoint! function copyto!(xs::AbstractVector, ys::Tuple)
    xs_ = copy(xs)
    copyto!(xs, ys), function (dxs)
        copyto!(xs_, xs)
        return (nothing, Tuple(dxs))
    end
end


end # module
