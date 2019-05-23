function rotgrad(::Type{T}, rb::RotationGate{N}) where {N, T}
    -sin(rb.theta / 2)/2 * IMatrix{1<<N}() + im/2 * cos(rb.theta / 2) * conj(mat(T, rb.block))
end

@adjoint function mat(::Type{T}, rb::RotationGate{N, RT}) where {T, N, RT}
    mat(T, rb), adjy -> (nothing, projection(rb.theta, sum(adjy .* rotgrad(T, rb))),)
end

@adjoint function mat(::Type{T}, A::GeneralMatrixBlock) where T
    mat(T, A), adjy -> (nothing, projection(A.mat, adjy))
end

function tegrad(::Type{T}, te::TimeEvolution, y) where T
    -im*(mat(te.H.block)*y)
end

@adjoint function mat(::Type{T}, te::TimeEvolution) where T
    y = mat(T, te)
    y, adjy -> (nothing, projection(te.dt, sum(adjy .* tegrad(T, te, y))),)
end

@adjoint function mat(::Type{T}, rb::Union{PutBlock{N, C, RT}, RT}) where {T, N, C, RT<:ConstGate.ConstantGate}
    mat(T, rb), adjy -> (nothing, nothing)
end

@adjoint function RotationGate(G, θ)
    RotationGate(G, θ), adjy->(nothing, adjy)
end

@adjoint function TimeEvolution(H, dt)
    TimeEvolution(H, dt), adjy->(nothing, adjy)
end

@adjoint function PutBlock{N}(block::GT, locs::NTuple{C, Int}) where {N, M, C, GT <: AbstractBlock{M}}
    PutBlock{N}(block, locs), adjy->(adjy.content, nothing)
end

@adjoint function ChainBlock(blocks) where N
    ChainBlock(blocks),
    adjy -> (adjy.blocks,)
end

# change the behavior later!
@adjoint function GeneralMatrixBlock{M, N}(m) where {M, N, T, MT <: AbstractMatrix{T}}
    GeneralMatrixBlock{M, N}(m), adjy->(adjy,)
end

@adjoint function GeneralMatrixBlock(m)
    GeneralMatrixBlock(m), adjy->(adjy,)
end

@adjoint function chain(blocks) where N
    chain(blocks),
    adjy -> (adjy.blocks,)
end

@adjoint function KronBlock{N, MT}(slots::Vector{Int}, locs::Vector{Int}, blocks::Vector{MT}) where {N, MT<:AbstractBlock}
    KronBlock{N, MT}(slots, locs, blocks),
    adjy -> (nothing, nothing, adjy.blocks)
end

@adjoint function KronBlock{N}(locs::Vector{Int}, blocks::Vector{MT}) where {N, MT<:AbstractBlock}
    KronBlock{N}(locs, blocks),
    adjy->(nothing, adjy.blocks)
end

@adjoint function Base.kron(total::Int, blocks::Pair{Int, <:AbstractBlock}...)
    kron(total, blocks...), adjy->(nothing, (nothing=>blk for blk in adjy.blocks)...)
end

@adjoint function ControlBlock{N, BT, C, M}(ctrl_locs, ctrl_config, block, locs) where {N, C, M, BT<:AbstractBlock}
    ControlBlock{N, BT, C, M}(ctrl_locs, ctrl_config, block, locs),
    adjy->(nothing, nothing, adjy.content, nothing)
end

@adjoint function YaoBlocks.cunmat(nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    y = YaoBlocks.cunmat(nbit, cbits, cvals, U0, locs)
    y, adjy-> (nothing, nothing, nothing, adjcunmat(y, adjy, nbit, cbits, cvals, U0, locs), nothing)
end

@adjoint function YaoBlocks.u1mat(nbit::Int, U1::SDMatrix, ibit::Int)
    y = YaoBlocks.u1mat(nbit, U1, ibit)
    y, adjy-> (nothing, adju1mat(y, adjy, nbit, U1, ibit), nothing)
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
    @inbounds U.vals .+= mat.vals[locs]
    return U
end

function _prepair_kronmat(k::KronBlock{N}) where N
    sizes = map(nqubits, subblocks(k))
    start_locs = @. N - $(k.locs) - sizes + 1

    order = sortperm(start_locs)
    sorted_start_locs = start_locs[order]
    num_bit_list = vcat(diff(push!(sorted_start_locs, N)) .- sizes[order])
    return order, num_bit_list, sorted_start_locs
end

@adjoint _prepair_kronmat(k::KronBlock) = _prepair_kronmat(k), adjy->nothing

function YaoBlocks.mat(::Type{T}, k::KronBlock{N}) where {T, N}
    order, num_bit_list, sorted_start_locs = _prepair_kronmat(k)
    blocks = subblocks(k)[order]
    return reduce(zip(blocks, num_bit_list), init=IMatrix{1 << sorted_start_locs[1], T}()) do x, y
        kron(x, mat(T, y[1]), IMatrix(1<<y[2]))
    end
end
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

@inline function adju1ij!(csc::SparseMatrixCSC{T}, i::Int,j::Int, adjU::Matrix) where T
    @inbounds begin
        adjU[1,1] += csc.nzval[2*i-1]
        adjU[1,2] += csc.nzval[2*i]
        adjU[2,1] += csc.nzval[2*j-1]
        adjU[2,2] += csc.nzval[2*j]
    end
    adjU
end

function adju1mat(y::AbstractMatrix, adjy, nbit::Int, U1::SDMatrix, ibit::Int)
    mask = bmask(ibit)
    step = 1<<(ibit-1)
    step_2 = 1<<ibit

    adjy = _render_adjy(adjy, y)
    adjU = _render_adjU(U1)

    for j = 0:step_2:1<<nbit-step
        @inbounds @simd for i = j+1:j+step
            adju1ij!(adjy, i, i+step, adjU)
        end
    end
    adjU
end

_render_adjy(adjy, y) = projection(y, adjy)

_render_adjU(U0::AbstractMatrix{T}) where T = zeros(T, size(U0)...)
_render_adjU(U0::SDSparseMatrixCSC{T}) where T = SparseMatrixCSC(size(U0)..., dynamicize(U0.colptr), dynamicize(U0.rowval), zeros(T, U0.nzval|>length))
_render_adjU(U0::SDDiagonal{T}) where T = Diagonal(zeros(T, size(U0, 1)))
_render_adjU(U0::SDPermMatrix{T}) where T = PermMatrix(U0.perm, zeros(T, length(U0.vals)))

function collect_gradients(st, out=Any[])
    for blk in st
        collect_gradients(blk, out)
    end
    out
end

collect_gradients(st::Number, out=any[]) = push!(out, st)
collect_gradients(st::Nothing, out=any[]) = out

@adjoint YaoBlocks.decode_sign(args...) = YaoBlocks.decode_sign(args...), adjy->nothing
