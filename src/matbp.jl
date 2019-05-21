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
    function l1(params::Vector)
        dispatch!(circuit, params)
        (v'* mat(circuit) * v)[] |> real
    end
    gradient_check(l1, [0.3, 0.6])
end

v = randn(ComplexF64, 4)
reg = rand_state(4)

cnot_mat = mat(control(2, 1, 2=>X))
function loss3(params::Vector)
    dispatch!(circuit, params)
    M = mat(circuit) |> Matrix
    norm(M-cnot_mat)
end

@adjoint function norm(x)
    y = norm(x)
    y, adjy->(adjy/y*x,)
end

gradient_check(norm, randn(3,3))

loss3([0.3, 0.1])
loss3'([0.3, 0.2])

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
