using Yao
using YaoBlocks: ConstGate
import Yao: apply!, ArrayReg, statevec, RotationGate
using Zygote
using Zygote: gradient, @adjoint
using LuxurySparse, SparseArrays, LinearAlgebra
using LinalgBackwards
using Random
using BitBasis: controller, controldo
using TupleTools

function rotgrad(::Type{T}, rb::RotationGate{N}) where {N, T}
    -sin(rb.theta / 2)/2 * IMatrix{1<<N}() + im/2 * cos(rb.theta / 2) * mat(T, rb.block)
end

@adjoint function mat(::Type{T}, rb::RotationGate{N, RT}) where {T, N, RT}
    mat(T, rb), adjy -> (nothing, RT(real(sum(adjy .* rotgrad(T, rb)))),)
end

@adjoint function RotationGate(G, θ)
    RotationGate(G, θ), adjy->(nothing, adjy)
end

@adjoint function PutBlock{N}(block::GT, locs::NTuple{C, Int}) where {N, M, C, GT <: AbstractBlock{M}}
    PutBlock{N}(block, locs), adjy->(adjy.content, adjy.locs)
end

@adjoint function ChainBlock(blocks) where N
    ChainBlock(blocks),
    adjy -> (adjy.blocks,)
end

@adjoint function KronBlock{N, MT}(slots::Vector{Int}, locs::Vector{Int}, blocks::Vector{MT}) where {N, MT<:AbstractBlock}
    KronBlock{N, MT}(slots, locs, blocks),
    adjy -> (adjy.slots, adjy.locs, adjy.blocks)
end

@adjoint function ControlBlock{N, BT, C, M}(ctrl_locs, ctrl_config, block, locs) where {N, C, M, BT<:AbstractBlock}
    ControlBlock{N, BT, C, M}(ctrl_locs, ctrl_config, block, locs),
    adjy->(adjy.ctrl_locs, adjy.ctrl_config, adjy.content, adjy.locs)
end

# data projection
@adjoint function *(sp::Union{SDSparseMatrixCSC, SDDiagonal, SDPermMatrix}, v::AbstractVector)
    sp*v, adjy -> (outer_projection(sp, adjy, v'), sp'*adjy)
end

@adjoint YaoBlocks.decode_sign(args...) = YaoBlocks.decode_sign(args...), adjy->nothing

@adjoint function *(v::LinearAlgebra.Adjoint{T, V}, sp::Union{SDSparseMatrixCSC, SDDiagonal, SDPermMatrix}) where {T, V<:AbstractVector}
    v*sp, adjy -> (adjy*sp', outer_projection(sp, v', adjy))
end

@adjoint function *(v::LinearAlgebra.Adjoint{T, V}, sp::SDDiagonal, v2::AbstractVector) where {T, V<:AbstractVector}
    v*sp, adjy -> (adjy*(sp*v2)', adjy*projection(sp, v', v2'), adjy*(v*sp)')
end

function outer_projection(y::SDSparseMatrixCSC, adjy, v)
    # adjy*v^T
    out = zero(y)
    is, js, vs = findnz(y)
    for (k,(i,j)) in enumerate(zip(is, js))
        @inbounds out.nzval[k] = adjy[i]*v[j]
    end
    out
end

outer_projection(y::SDDiagonal, adjy, v) = Diagonal(adjy.*v)

"""
Project a dense matrix to a sparse matrix
"""
function projection(y::AbstractSparseMatrix, m::AbstractMatrix)
    out = zero(y)
    is, js, vs = findnz(y)
    for (k,(i,j)) in enumerate(zip(is, js))
        @inbounds out.nzval[k] = m[i,j]
    end
    out
end

projection(y::SDDiagonal, m::AbstractMatrix) = Diagonal(diag(m))

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

@inline function adjunij!(mat::SparseMatrixCSC, locs, U::Matrix)
    for j = 1:size(U, 2)
        @inbounds adjsetcol!(mat, locs[j], locs, view(U,:,j))
    end
    return U
end

@inline function adjunij!(mat::SparseMatrixCSC, locs, U::SparseMatrixCSC)
    for j = 1:size(U, 2)
        S = U.colptr[j]
        E = U.colptr[j+1]-1
        @inbounds adjsetcol!(mat, locs[j], locs, view(U.nzval,S:E))
    end
    return U
end

@inline function adjunij!(mat::SDDiagonal, locs, U::Diagonal)
    @inbounds U.diag .+= mat.diag[locs]
    return U
end

@inline function adjunij!(mat::SDPermMatrix, locs, U::PermMatrix)
    @inbounds U.vals .+= pm.vals[locs]
    return U
end

#=
function YaoBlocks.mat(::Type{T}, k::KronBlock{N}) where {T, N}
    sizes = map(nqubits, subblocks(k))
    start_locs = @. N - $(k.locs) - sizes + 1

    order = sortperm(start_locs)
    sorted_start_locs = start_locs[order]
    num_bit_list = vcat(diff(push!(sorted_start_locs, N)) .- sizes[order])

    blocks = subblocks(k)[order]
    return _kron_mat(T, blocks, num_bit_list, sorted_start_locs)
end

function _kron_mat(T, blocks, num_bit_list, sorted_start_locs)
    return reduce(zip(blocks, num_bit_list), init=IMatrix{1 << sorted_start_locs[1], T}()) do x, y
        kron(x, mat(T, y[1]), IMatrix(1<<y[2]))
    end
end
=#
#@adjoint function _kron_mat(::Type{T}, k::KronBlock{N}) where {T, N}
#    y, back = forward(f, args...)
#    return back(sensitivity(y))
#end

@adjoint function Base.Iterators.Zip(tp)
    Base.Iterators.Zip(tp), adjy-> ((@show adjy;zip(adjy...)),)
end

@adjoint function reduce(func, xs; kwargs...)
    backs = Any[]
    ys = Any[]
    function nfunc(x, x2)
        y, back = forward(func, x, x2)
        push!(backs, back)
        push!(ys, y)
        return y
    end
    reduce(nfunc, xs; kwargs...),
    function (adjy)
        res = Vector{Any}(undef, length(ys))
        for i=length(ys):-1:1
            back, y = backs[i], ys[i]
            adjy, res[i] = back(adjy)
        end
        if !haskey(kwargs, :init)
            insert!(res, 1, adjy)
        end
        return (nothing, res)
    end
end

@adjoint function mapreduce(op, func, xs; kwargs...)
    opbacks = Any[]
    backs = Any[]
    ys = Any[]
    function nop(x)
        y, back = forward(op,x)
        push!(opbacks, back)
        y
    end
    function nfunc(x, x2)
        y, back = forward(func, x, x2)
        push!(backs, back)
        push!(ys, y)
        return y
    end
    mapreduce(nop, nfunc, xs; kwargs...),
    function (adjy)
        offset = haskey(kwargs, :init) ? 0 : 1
        res = Vector{Any}(undef, length(ys)+offset)
        for i=length(ys):-1:1
            opback, back, y = opbacks[i+offset], backs[i], ys[i]
            adjy, adjthis = back(adjy)
            res[i+offset], = opback(adjthis)
        end
        if offset==1
            res[1], = opbacks[1](adjy)
        end
        return (nothing, nothing, res)
    end
end

@adjoint function collect(::Type{T}, source::TS) where {T, TS}
    collect(T, source),
    adjy -> (adjy,)   # adjy -> convert(TS, adjy)
end

@adjoint Iterators.reverse(x::T) where T = Iterators.reverse(x), adjy->(collect(Iterators.reverse(adjy)),)  # convert is better

function adjcunmat(adjy::AbstractMatrix, nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix{T}, locs::NTuple{M, Int}) where {C, M, T}
    U, ic, locs_raw = YaoBlocks.reorder_unitary(nbit, cbits, cvals, U0, locs)
    adjU = _render_adjU(U)

    ctest = controller(cbits, cvals)

    controldo(ic) do i
        adjunij!(adjy, locs_raw+i, adjU)
    end

    adjU = all(TupleTools.diff(locs).>0) ? adjU : YaoBase.reorder(adjU, collect(locs)|>sortperm|>sortperm)
    adjU
end

Base.zero(pm::PermMatrix) = PermMatrix(pm.perm, zero(pm.vals))

_render_adjU(U0::AbstractMatrix{T}) where T = zeros(T, size(U0)...)
_render_adjU(U0::SDSparseMatrixCSC{T}) where T = SparseMatrixCSC(size(U0)..., dynamicize(U0.colptr), dynamicize(U0.rowval), zeros(T, U0.nzval|>length))
_render_adjU(U0::SDDiagonal{T}) where T = Diagonal(zeros(T, size(U0, 1)))
_render_adjU(U0::SDPermMatrix{T}) where T = PermMatrix(U0.perm, zero(U0.vals))

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
