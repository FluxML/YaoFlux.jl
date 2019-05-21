include("adjbase.jl")
include("adjYao.jl")
using Test
@testset "mat grad" begin
    Random.seed!(5)
    # rotation block
    ng(f, θ, δ=1e-5) = (f(θ+δ/2) - f(θ-δ/2))/δ
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

    nbit = 5
    θ = 0.5
    b = randn(ComplexF64, 1<<nbit)
    gz(x) = (b'*(mat(put(nbit, 2=>Rz(x)))*b))[] |> real
    @test isapprox(gz'(θ), ng(gz, θ), atol=1e-4)
    gx(x) = (b'*(mat(put(nbit, 2=>Rx(x)))*b))[] |> real
    @test isapprox(gx'(θ), ng(gx, θ), atol=1e-4)

    # control block
    cgn(x) = (b'*(mat(control(nbit, 3, (4,2)=>rot(ConstGate.CNOT, x)))*b))[] |> real
    @test isapprox(cgn'(θ), ng(cgn, θ), atol=1e-4)

    # Chain Block
    b = randn(ComplexF64, 2)
    gctrl(x) = (b'*mat(chain([Rx(x),Ry(x+0.4)]))*b)[] |> real   # collect not correctly defined
    @show gctrl(0.5)
    @test isapprox(gctrl'(θ), ng(gctrl, θ), atol=1e-4)

    # Kron Block
    b = randn(ComplexF64, 4)
    gkron(x) = (b'*mat(kron(Rx(x),Ry(x+0.4)))*b)[] |> real
    @show gkron(0.5)
    #@test isapprox(gkron'(θ), ng(gkron, θ), atol=1e-4)

    v = randn(ComplexF64, 4)
    circuit = chain(2, [put(2, 2=>Rx(0.0)), control(2, 1, 2=>Z), put(2, 2=>Rz(0.0))])
    dispatch!(circuit, [0.4, 0.6])
    function l1(circuit)
        (v'* mat(circuit) * v)[] |> real
    end
    @show l1'(circuit)
    function loss1(params)
        dispatch!(circuit, params)
        (v'* mat(circuit) * v)[] |> real
    end
    @test gradient_check(loss1, [0.4, 0.6])
end

#v = randn(ComplexF64, 4)
#circuit = chain(2, [put(2, 2=>chain([Rz(0.0), Rx(0.0)])), control(2, 1, 2=>Z), put(2, 2=>chain([Rx(0.0), Rz(0.0)]))])
#circuit = chain(2, [put(2, 1=>H), put(2, 2=>Rz(0.0)), put(2, 2=>Rx(0.0)), control(2, 1, 2=>Z), put(2, 2=>Rx(0.0)), put(2, 2=>Rz(0.0))])
circuit = chain(2, [put(2, 2=>Rz(0.0)), put(2, 2=>Rx(0.0)), control(2, 1, 2=>Z), put(2, 2=>Rx(0.0)), put(2, 2=>Rz(0.0))])
using QuAlgorithmZoo: heisenberg
h = mat(heisenberg(2))
v0 = statevec(zero_state(2))
function energy4(params::Vector)
    dispatch!(circuit, params)
    v = mat(circuit) * v0
    (v'* h * v)[] |> real
end

energy4([0.3, 0.6, 0.3, 0.2])
energy4'([0.3, 0.6, 0.3, 0.2])

using Zygote: @adjoint!, grad_mut, _forward
function _forward(cx::Context, ::typeof(collect), g::Base.Generator)

@adjoint! function dispatch!(circuit, params)
    dispatch!(circuit, params),
    function (x_)
        dstk = grad_mut(__context__, circuit)
        @show dstk
        dstk = grad_mut(__context__, params)
        @show dstk
        @show x_
        (nothing, grad)
    end
end

gradient_check(l1, [0.3, 0.6])

v = randn(ComplexF64, 4)
reg = rand_state(4)

cnot_mat = mat(control(2, 1, 2=>X))
function loss4(params::Vector)
    dispatch!(circuit, params)
    M = mat(circuit)
    norm(M-cnot_mat)
end

@adjoint invperm(x) = invperm(x), adjy -> nothing
@adjoint function norm(x)
    y = norm(x)
    y, adjy->(adjy/y*x,)
end

gradient_check(norm, randn(ComplexF64, 3,3))

loss4([0.3, 0.1, 0.3, 0.5])
loss4'([0.3, 0.2, 0.3, 0.5])

norm
function l2(A)
    norm(A - B)
end

@testset "csc mul" begin
    Random.seed!(2)
    T = ComplexF64
    v = randn(T, 4)
    A = sprand(T, 4,4,0.5)
    f(v) = (v'*(A*v))[] |> real
    g(A) = (v'*(A*v))[] |> real
    @test gradient_check(f, v)
    @test g'(A) ≈ projection(A, g'(Matrix(A)))

    f2(v) = (v'*A*v)[] |> real
    g2(A) = (v'*A*v)[] |> real
    @test gradient_check(f2, v)
    @test g2'(A) ≈ projection(A, g2'(Matrix(A)))
end

@testset "diag mul" begin
    Random.seed!(2)
    T = ComplexF64
    v = randn(T, 4)
    A = Diagonal(randn(T, 4))
    f(v) = (v'*(A*v))[] |> real
    g(A) = (v'*(A*v))[] |> real
    @test gradient_check(f, v)
    @test g'(A) ≈ projection(A, g'(Matrix(A)))

    f2(v) = ((v'*A)*v)[] |> real
    g2(A) = (v'*A*v)[] |> real  # this does not pass?
    @test gradient_check(f2, v)
    #@test g2'(A) ≈ projection(A, g2'(Matrix(A)))
end

@testset "reduce" begin
    f3(x) = reduce(+, x, init=3.0)
    @test f3'([1,2.0,5]) == [1,1,1.]
    f4(x) = mapreduce(x->x^2, +, x, init=3.0)
    @test f4'([1,2.0,5]) == [2,4,10.]
    f5(x) = reduce(+, x)
    @test f5'([1,2.0,5]) == [1,1,1.]
    f6(x) = mapreduce(x->x^2, +, x)
    @test f6'([1,2.0,5]) == [2,4,10.]
end

@testset "zip" begin
    ys = [6,5,4]
    function f4(xs)
        reduce((pre,this)->pre+this[1]*this[2], zip(xs, ys), init=0)
    end
    @test f4'([1,2,3]) |> collect == [6,5,4]
end

@testset "projection" begin
    y = pmrand(6)
    @test projection(y, y) == y
    y = sprand(6, 6, 0.5)
    @test projection(y, y) == y
    y = Diagonal(randn(6))
    @test projection(y, y) == y
end
