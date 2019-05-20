using Yao
using YaoBlocks: ConstGate
import Yao: apply!, ArrayReg, statevec, RotationGate
using Zygote
using Zygote: gradient, @adjoint
using LuxurySparse, SparseArrays, LinearAlgebra
using LinalgBackwards
using Random

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

# data projection
@adjoint function *(sp::Union{SDSparseMatrixCSC, SDDiagonal, SDPermMatrix}, v::AbstractVector)
    sp*v, adjy -> (outer_projection(sp, adjy, v'), sp'*adjy)
end

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
