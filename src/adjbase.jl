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

Base.zero(pm::PermMatrix) = PermMatrix(pm.perm, zero(pm.vals))
projection(y::SDDiagonal, m::AbstractMatrix) = Diagonal(diag(m))
function projection(y::PermMatrix, m::AbstractMatrix)
    res = zero(y)
    for i=1:size(res, 1)
        @inbounds res.vals[i] = m[i,res.perm[i]]
    end
    res
end
projection(x::RT, adjx::Complex) where RT<:Real = RT(real(adjx))
