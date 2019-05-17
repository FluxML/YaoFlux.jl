using FillArrays
using YaoFlux, Yao
using Test, LuxurySparse, Zygote, YaoBase
using SparseArrays
import Zygote: @adjoint

@adjoint SparseArrays.SparseMatrixCSC(A::PermMatrix) = SparseMatrixCSC(A), Δ->(Δ, )

function rotmat(x)
    I = IMatrix{2, ComplexF64}()
    return I * cos(x/2) - im * sin(x / 2) * Const.X
end

Zygote.refresh()
_, back = Zygote._forward(rotmat, 0.1)
back(rand(2, 2))


_, back = Zygote._forward(sprand(2, 2, 0.1), sprand(2, 2, 0.2)) do x, y
    x - y
end

rotmat(0.1)

@which IMatrix{2, ComplexF64}() * 0.1 - Const.X

SparseMatrixCSC()

t = IMatrix{2, ComplexF64}() * 0.1

@which SparseMatrixCSC(t)

Zygote.refresh()
f(x) = x * Const.X
_, back = Zygote._forward(f, 0.1)
back(ComplexF64[0 1;1 0])


f(x) = real(sum(x * Const.X))
f'(0.1)


using InteractiveUtils
@edit IMatrix{2, Float64}() * cos(0.1)


using LinearAlgebra

Zygote.refresh()
_, back = Zygote._forward(0.1) do x
    Diagonal(Fill(x, 2))
end

back(rand(2, 2))

Zygote.@adjoint (::Type{T})(x::Number, sz) where {T <: Fill} = Fill(x, sz), Δ -> (sum(Δ), nothing)


function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

gradcheck(f, xs...) =
  all(isapprox.(ngradient(f, xs...),
                gradient(f, xs...), rtol = 1e-5, atol = 1e-5))

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)

using Zygote: gradient


gradcheck(x->sum(Fill(x[], (2, 2))), [0.1])

ngradient(x->sum(Fill(x[], (2, 2))), [0.1])



circuit(ps) = chain(Rx(ps[1]), Rz(ps[2]), Rx(ps[3]))

