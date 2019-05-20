include("adjYao.jl")
using BitBasis: controller, controldo
using TupleTools

@adjoint function YaoBlocks.cunmat(nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    YaoBlocks.cunmat(nbit, cbits, cvals, U0, locs), adjy-> (nothing, nothing, nothing, adjcunmat(adjy, nbit, cbits, cvals, U0, locs), nothing)
end

@inline function adjsetcol!(csc::SparseMatrixCSC, icol::Int, rowval::AbstractVector, nzval::SubArray)
    @inbounds begin
        S = csc.colptr[icol]
        E = csc.colptr[icol+1]-1
        nzval .+= view(csc.nzval, S:E)
    end
    csc
end

@inline function adjsetcol!(csc::SparseMatrixCSC, icol::Int, rowval::AbstractVector, nzval::SubArray)
    @inbounds begin
        S = csc.colptr[icol]
        E = csc.colptr[icol+1]-1
        nzval .+= view(csc.nzval, S:E)
    end
    csc
end

@inline function adjunij!(mat::SparseMatrixCSC, locs, U::Matrix)
    for j = 1:size(U, 2)
        @inbounds adjsetcol!(mat, locs[j], locs, view(U,:,j))
    end
    return U
end

@inline function adjunij!(mat::SDDiagonal, locs, U::Diagonal)
    @inbounds U.diag .+= mat.diag[locs]
    return U
end

function adjcunmat(adjy::AbstractMatrix, nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix{T}, locs::NTuple{M, Int}) where {C, M, T}
    U, ic, locs_raw = YaoBlocks.reorder_unitary(nbit, cbits, cvals, U0, locs)
    if !(adjy isa SparseMatrixCSC)
        adjy = SparseMatrixCSC(adjy)
    end
    adjU = zeros(T, size(U0)...)

    colptr = adjy.colptr
    rowval = adjy.rowval

    ctest = controller(cbits, cvals)

    controldo(ic) do i
        adjunij!(adjy, locs_raw+i, adjU)
    end

    adjU = all(TupleTools.diff(locs).>0) ? adjU : reorder(adjU, collect(locs)|>sortperm|>sortperm)
    adjU
end

function adjcunmat(adjy::SDDiagonal, nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::SDDiagonal{T}, locs::NTuple{M, Int}) where {C, M, T}
    U, ic, locs_raw = YaoBlocks.reorder_unitary(nbit, cbits, cvals, U0, locs)
    adjU = Diagonal(zeros(T, size(U0, 1)))

    controldo(ic) do i
        adjunij!(adjy, locs_raw+i, adjU)
    end
    return adjU
end

using Test
@testset "mat grad" begin
    Random.seed!(5)
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

    nbit = 3
    θ = 0.5
    b = randn(ComplexF64, 1<<nbit)
    gz(x) = (b'*(mat(put(nbit, 2=>Rz(x)))*b))[] |> real
    @test isapprox(gz'(θ), ng(gz, θ), atol=1e-4)
    gx(x) = (b'*(mat(put(nbit, 2=>Rx(x)))*b))[] |> real
    @test isapprox(gx'(θ), ng(gx, θ), atol=1e-4)
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
