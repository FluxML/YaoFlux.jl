using Test
using Yao, Zygote
using YaoBlocks: ConstGate
using YaoFlux, Random

ng(f, θ, δ=1e-5) = (f(θ+δ/2) - f(θ-δ/2))/δ

@testset "rot mat grad" begin
    Random.seed!(5)
    # rotation block
    gg(x::Float64) = sum(mat(ComplexF64, Rx(x))) |> real
    @test isapprox(gg'(0.5), ng(gg, 0.5), atol=1e-4)
    gy(x::Float64) = sum(mat(Rz(x))) |> real
    @test isapprox(gy'(0.5), ng(gy, 0.5), atol=1e-4)

    rd = randn(ComplexF64, 4,4)
    A = randn(ComplexF64, 4,4)
    gt(x) = sum((x*A)*rd) |> real
    @test isapprox(gt'(0.5) |> real, ng(gt, 0.5), atol=1e-4)
    gcnot(x::Float64) = sum(mat(rot(ConstGate.CNOT, x))*rd) |> real
    @test isapprox(gcnot'(0.5), ng(gcnot, 0.5), atol=1e-4)
end

@testset "put mat grad" begin
    nbit = 5
    θ = 0.5
    Random.seed!(1)
    b = randn(ComplexF64, 1<<nbit)
    gz(x) = (b'*(mat(put(nbit, 2=>Rz(x)))*b))[] |> real
    @test isapprox(gz'(θ), ng(gz, θ), atol=1e-4)
    gx(x) = (b'*(mat(put(nbit, 2=>Rx(x)))*b))[] |> real
    @test isapprox(gx'(θ), ng(gx, θ), atol=1e-4)

    ru = rand_hermitian(2)
    gm(x) = (b'*(mat(put(nbit, 2=>matblock(x)))*b))[] |> real
    @test gradient_check(gm, ru)

    gd(x) = (b'*(mat(put(nbit, 2=>time_evolve(matblock(ru), x)))*b))[] |> real
    @test isapprox(gd'(θ), ng(gd, θ), atol=1e-4)
    gd(x) = (b'*(mat(put(nbit, 2=>time_evolve(matblock(ru), x)))*b))[] |> real
    @test isapprox(gd'(im*θ), ng(gd, im*θ, 1e-5im), atol=1e-4)
end

@testset "control mat grad" begin
    nbit = 5
    θ = 0.5
    Random.seed!(5)
    b = randn(ComplexF64, 1<<nbit)
    # control block
    cgn(x) = (b'*(mat(control(nbit, 3, (4,2)=>rot(ConstGate.CNOT, x)))*b))[] |> real
    @test isapprox(cgn'(θ), ng(cgn, θ), atol=1e-4)
end

@testset "chain mat grad" begin
    nbit = 5
    θ = 0.5
    # Chain Block
    Random.seed!(5)
    b = randn(ComplexF64, 2)
    gctrl(x) = (b'*mat(chain([Rx(x),Ry(x+0.4)]))*b)[] |> real   # collect not correctly defined
    @test isapprox(gctrl'(θ), ng(gctrl, θ), atol=1e-4)
end

@testset "general mat grad" begin
    v = randn(ComplexF64, 4)
    circuit = chain(2, [put(2, 2=>Rx(0.0)), control(2, 1, 2=>Z), put(2, 2=>Rz(0.0))])
    dispatch!(circuit, [0.4, 0.6])
    function l1(circuit)
        (v'* mat(circuit) * v)[] |> real
    end
    g1 = x->(dispatch!(circuit, [x, 0.6]); l1(circuit))
    g2 = x->(dispatch!(circuit, [0.4, x]); l1(circuit))
    @test collect_gradients(l1'(circuit)) ≈ [ng(g1, 0.4), ng(g2, 0.6)]
    function loss1(params)
        dispatch!(circuit, params)   # dispatch! will fail for deep models!!!!!
        (v'* mat(circuit) * v)[] |> real
    end
    @test_broken gradient_check(loss1, [0.4, 0.6])
end

@testset "kron mat grad" begin
    Random.seed!(2)
    nbit = 2
    θ = 1.2
    # Kron Block
    b = randn(ComplexF64, 1<<nbit)
    gkron(x) = (b'*mat(kron(nbit, 1=>Rx(x), 2=>Rz(x+0.8)))*b)[] |> real
    @test isapprox(gkron'(θ), ng(gkron, θ), atol=1e-5)

    nbit = 5
    b = randn(ComplexF64, 1<<nbit)
    gkron2(x) = (b'*mat(kron(nbit, 4=>Rx(x), 1=>Rz(x+0.8)))*b)[] |> real
    @test isapprox(gkron2'(θ), ng(gkron2, θ), atol=1e-5)

    nbit = 5
    b = randn(ComplexF64, 1<<nbit)
    gkron3(x) = (b'*mat(kron(nbit, 4=>Rx(x), 1=>Ry(x+0.4)))*b)[] |> real
    @test isapprox(gkron3'(θ), ng(gkron3, θ), atol=1e-5)
end
