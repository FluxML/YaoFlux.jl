module YaoFlux

using Zygote, LuxurySparse, Yao, YaoBase, SparseArrays, BitBasis
import Zygote: Context

Zygote.@adjoint (::Type{T})(perms, vals) where T <: PermMatrix = T(perms, vals), Δ -> nothing
Zygote.@adjoint (::Type{T})() where T <: IMatrix = T(), Δ -> nothing
Zygote.@adjoint Base.:(*)(A::Number, B::PermMatrix) = A * B, Δ->(sum(Δ .* B), A * Δ)
Zygote.@adjoint Base.:(*)(A::PermMatrix, B::Number) = A * B, Δ->(A * Δ, sum(Δ .* B))

Zygote.@adjoint SparseArrays.SparseMatrixCSC(A::PermMatrix) = SparseMatrixCSC(A), Δ->(Δ, )

Zygote.@adjoint BitBasis.onehot(::Type{T}, nbits::Int, x::Integer, nbatch::Int) where T = onehot(T, nbits, x, nbatch), Δ -> nothing

# YaoArrayRegister

# struct GradReg{Reg, G}
#     r::Reg # this is the register to back prop
#     grad::G # gradient of the register
# end

# no gradient if meets a constructor
# Zygote.@adjoint (::Type{T})(r) where T <: ArrayReg = T(r), Δ->nothing

# Zygote.@adjoint function YaoArrayRegister.state(r::ArrayReg)
#     state(r), Δ -> (GradReg(r, Δ), )
# end

Zygote.@adjoint function Base.getfield(r::ArrayReg, n::Symbol)
    getfield(r, n), Δ -> (ArrayReg(Δ), nothing)
end

# YaoBlocks

# Zygote.@adjoint (::Type{T})(P::AbstractBlock, θ) where T <: RotationGate = T(P, θ), Δ->(nothing, real(Δ),)

# U * r
# delta * r', U' * delta
# in most cases block do not have parameters
# Zygote.@adjoint function YaoBlocks.apply!(r::ArrayReg, x::ConstantGate)
#     apply!(r, x), function (Δ)
#         apply!(copy(Δ), x'), nothing # no grad for constant gates
#     end
# end

# Zygote.@adjoint function YaoBlocks.apply!(r::ArrayReg{B, T}, x::RotationGate) where {B, T}
#     r_ = copy(r)
#     apply!(r, x), function (Δ)
#         # dU = reshape(Δ * state(r)', size(U))
#         # dr = reshape(U' * Δ, size(U))
#         # apply!(Δ, x')
#         dispatch!(-, x, π)
#         apply!(r_, -x/2)
#         dispatch!(+, x, π)

#         return (apply!(copy(Δ), x'), sum(state(Δ) * state(r_)'))
#     end
# end

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
