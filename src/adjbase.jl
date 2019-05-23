Zygote.@adjoint (::Type{T})(perms, vals) where T <: PermMatrix = T(perms, vals), Δ -> nothing
Zygote.@adjoint (::Type{T})() where T <: IMatrix = T(), Δ -> nothing
Zygote.@adjoint Base.:(*)(A::Number, B::PermMatrix) = A * B, Δ->(sum(Δ .* B), A * Δ)
Zygote.@adjoint Base.:(*)(A::PermMatrix, B::Number) = A * B, Δ->(A * Δ, sum(Δ .* B))

Zygote.@adjoint SparseArrays.SparseMatrixCSC(A::PermMatrix) = SparseMatrixCSC(A), Δ->(Δ, )

Zygote.@adjoint BitBasis.onehot(::Type{T}, nbits::Int, x::Integer, nbatch::Int) where T = onehot(T, nbits, x, nbatch), Δ -> nothing

# upstreams
Zygote.@adjoint Base.:(-)(a, b) = a-b, Δ -> (Δ, -Δ)
Zygote.@adjoint Base.:(+)(a, b) = a+b, Δ -> (Δ, Δ)

# require mutate
Zygote.@adjoint! function copyto!(xs::AbstractVector, ys::Tuple)
    xs_ = copy(xs)
    copyto!(xs, ys), function (dxs)
        copyto!(xs_, xs)
        return (nothing, Tuple(dxs))
    end
end

@adjoint function Base.Iterators.Zip(tp)
    Base.Iterators.Zip(tp), adjy-> ([[x...] for x in zip(adjy...)],)
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
@adjoint Pair{T1, T2}(a, b) where {T1, T2} = Pair{T1, T2}(a, b), adjy->(adjy.first, adjy.second)

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
projection(x::T, y::T) where T = y
projection(x::T1, y::T2) where {T1, T2} = convert(T1, y)

# fix kron in Zygote if having time
#function kron_back(a::AbstractMatrix, b::AbstractMatrix, adjy)
#end
