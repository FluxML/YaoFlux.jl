function rotgrad(::Type{T}, rb::RotationGate{N}) where {N, T}
    -sin(rb.theta / 2)/2 * IMatrix{1<<N}() + im/2 * cos(rb.theta / 2) * mat(T, rb.block)
end

@adjoint function mat(::Type{T}, rb::RotationGate{N, RT}) where {T, N, RT}
    mat(T, rb), adjy -> (nothing, projection(rb.theta, sum(adjy .* rotgrad(T, rb))),)
end

@adjoint function mat(::Type{T}, rb::Union{PutBlock{N, C, RT}, RT}) where {T, N, C, RT<:ConstGate.ConstantGate}
    mat(T, rb), adjy -> (nothing, nothing)
end

@adjoint function RotationGate(G, θ)
    RotationGate(G, θ), adjy->(nothing, adjy)
end

@adjoint function PutBlock{N}(block::GT, locs::NTuple{C, Int}) where {N, M, C, GT <: AbstractBlock{M}}
    PutBlock{N}(block, locs), adjy->(adjy.content, adjy.locs)
end

@adjoint function ChainBlock(blocks) where N
    ChainBlock(blocks),
    adjy -> ((@show adjy.blocks; adjy.blocks),)
end

@adjoint function chain(blocks) where N
    chain(blocks),
    adjy -> ((@show adjy.blocks; adjy.blocks),)
end

@adjoint function KronBlock{N, MT}(slots::Vector{Int}, locs::Vector{Int}, blocks::Vector{MT}) where {N, MT<:AbstractBlock}
    KronBlock{N, MT}(slots, locs, blocks),
    adjy -> (adjy.slots, adjy.locs, adjy.blocks)
end

@adjoint function ControlBlock{N, BT, C, M}(ctrl_locs, ctrl_config, block, locs) where {N, C, M, BT<:AbstractBlock}
    ControlBlock{N, BT, C, M}(ctrl_locs, ctrl_config, block, locs),
    adjy->(adjy.ctrl_locs, adjy.ctrl_config, adjy.content, adjy.locs)
end

@adjoint function YaoBlocks.cunmat(nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    y = YaoBlocks.cunmat(nbit, cbits, cvals, U0, locs)
    y, adjy-> (nothing, nothing, nothing, adjcunmat(y, adjy, nbit, cbits, cvals, U0, locs), nothing)
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

function adjcunmat(y::AbstractMatrix, adjy::AbstractMatrix, nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix{T}, locs::NTuple{M, Int}) where {C, M, T}
    U, ic, locs_raw = YaoBlocks.reorder_unitary(nbit, cbits, cvals, U0, locs)
    adjy = _render_adjy(adjy, y)
    adjU = _render_adjU(U)

    ctest = controller(cbits, cvals)

    controldo(ic) do i
        adjunij!(adjy, locs_raw+i, adjU)
    end

    adjU = all(TupleTools.diff(locs).>0) ? adjU : YaoBase.reorder(adjU, collect(locs)|>sortperm|>sortperm)
    adjU
end
_render_adjy(adjy, y) = projection(y, adjy)

_render_adjU(U0::AbstractMatrix{T}) where T = zeros(T, size(U0)...)
_render_adjU(U0::SDSparseMatrixCSC{T}) where T = SparseMatrixCSC(size(U0)..., dynamicize(U0.colptr), dynamicize(U0.rowval), zeros(T, U0.nzval|>length))
_render_adjU(U0::SDDiagonal{T}) where T = Diagonal(zeros(T, size(U0, 1)))
_render_adjU(U0::SDPermMatrix{T}) where T = PermMatrix(U0.perm, zero(U0.vals))
