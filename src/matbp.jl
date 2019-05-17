using Yao
import Yao: apply!, ArrayReg, statevec, RotationGate
using Zygote
using Zygote: gradient, @adjoint
using LuxurySparse

function rotgrad(::Type{T}, rb::RotationGate{N}) where {N, T}
    -sin(rb.theta / 2)/2 * IMatrix{1<<N}() - im/2 * cos(rb.theta / 2) * mat(T, rb.block)
end

@adjoint function mat(::Type{T}, rb::RotationGate{N, RT}) where {T, N, RT}
    mat(T, rb), adjy -> (nothing, (@show adjy; RT(real(sum(adjy .* rotgrad(T, rb))))),)
end

@adjoint function RotationGate(G, θ)
    RotationGate(G, θ), adjy->(nothing, adjy)
end

using Test
@testset "mat grad" begin
    ng(f, θ, δ=1e-5) = (f(θ+δ/2) - f(θ-δ/2))/δ

    gg(x::Float64) = sum(mat(ComplexF64, Rx(x))) |> real
    @test isapprox(gg'(0.5), ng(gg, 0.5), atol=1e-4)
    gy(x::Float64) = sum(mat(Ry(x))) |> real
    @test isapprox(gy'(0.5), ng(gy, 0.5), atol=1e-4)
end
