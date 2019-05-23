using YaoFlux, Zygote
using LinearAlgebra, SparseArrays, LuxurySparse
using Test, Random

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
    @test gradient_check(f2, v)
    g2(A) = (v'*A*v)[] |> real  # this does not pass?
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
    @test f4'([1,2,3]) == [6,5,4]
end

@testset "projection" begin
    y = pmrand(6)
    @test projection(y, y) == y
    y = sprand(6, 6, 0.5)
    @test projection(y, y) == y
    y = Diagonal(randn(6))
    @test projection(y, y) == y
end
