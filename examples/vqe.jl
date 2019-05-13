using Yao
using LinearAlgebra
using Zygote
using Zygote: gradient
using Flux: Optimise, ADAM, Descent

function ngradient(f, x::AbstractVector; δ::Float64=1e-5)
    g = zero(x)
    for i=1:length(x)
        x[i] += δ/2
        f₊ = f(x)
        x[i] -= δ
        f₋ = f(x)
        x[i] += δ/2
        g[i] = (f₊ - f₋)/δ
    end
    g
end

"""get the hamiltonian of a two-site Hubbard model"""
function hubbard_hamiltonain(t::Real, U::Real)
    0.5t * (kron(X, Z, X, I2) +
        kron(Y, Z, Y, I2) +
        kron(I2, X, Z, X) +
        kron(I2, Y, Z, Y)) +
    0.25U * (kron(Z, Z, I2, I2) +
        kron(I2, I2, Z, Z))
end

function energy_function(H::AbstractMatrix)
    x::Vector -> real(x'*H*x/(x'*x))[]
end

function train!(efunc, x; optimizer=ADAM(0.1), niter::Int=100)
    for i = 1:niter
        Optimise.update!(optimizer, x, vec(efunc'(x)))
        println("Step $i, E = $(efunc(x))")
    end
    x
end

"""Variational Quantum Eigensolver"""
function energy_function!(H::AbstractBlock{N}, ansatz_circuit::AbstractBlock{N}) where N
    function energy(x::Vector)
        dispatch!(ansatz_circuit, x)
        real(expect(H, zero_state(N) |> ansatz_circuit))
    end
end

using Test

"""numeric differentiation version for testing"""
function ntrain!(efunc, x; optimizer=ADAM(0.1), niter::Int=100)
    for i = 1:niter
        Optimise.update!(optimizer, x, ngradient(efunc, x))
        println("Step $i, E = $(efunc(x))")
    end
    x
end

@testset "variational eigensolver" begin
    hami = hubbard_hamiltonain(1, 1)
    Hmat = mat(hami)
    E_exact = eigen(Matrix(Hmat)).values[1]
    println("Exact Energy is $(E_exact)")
    x = randn(ComplexF64, size(Hmat, 2))
    train!(energy_function(Hmat), x; optimizer=Descent(2.0), niter=100)
    @test isapprox(energy_function(Hmat)(x), E_exact; atol=1e-5)
end

using QuAlgorithmZoo: random_diff_circuit
@testset "variational quantum eigensolver - numeric grad" begin
    hami = hubbard_hamiltonain(1, 1)
    ansatz_circuit = random_diff_circuit(4, 2, [1=>2, 2=>3, 3=>4])
    Hmat = mat(hami)
    E_exact = eigen(Matrix(Hmat)).values[1]
    println("Exact Energy is $(E_exact)")
    x = 2π * rand(nparameters(ansatz_circuit))
    ntrain!(energy_function!(hami, ansatz_circuit), x; optimizer=ADAM(0.5), niter=100)
    @test isapprox(energy_function!(hami, ansatz_circuit)(x), E_exact; rtol=5e-2)
end

@testset "variational quantum eigensolver - zygote grad" begin
    hami = hubbard_hamiltonain(1, 1)
    ansatz_circuit = random_diff_circuit(4, 2, [1=>2, 2=>3, 3=>4])
    Hmat = mat(hami)
    E_exact = eigen(Matrix(Hmat)).values[1]
    println("Exact Energy is $(E_exact)")
    x = 2π * rand(nparameters(ansatz_circuit))
    train!(energy_function!(hami, ansatz_circuit), x; optimizer=ADAM(0.5), niter=100)
    @test isapprox(energy_function!(hami, ansatz_circuit)(x), E_exact; rtol=5e-2)
end
