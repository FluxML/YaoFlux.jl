module YaoFlux

using Zygote
using Zygote: gradient, @adjoint
using Zygote, LuxurySparse, Yao, YaoBase, SparseArrays, BitBasis
import Zygote: Context

using Yao
using YaoBlocks: ConstGate
import Yao: apply!, ArrayReg, statevec, RotationGate

using LuxurySparse, SparseArrays, LinearAlgebra
using BitBasis: controller, controldo
using TupleTools

export gradient_check, projection, collect_gradients

"""A safe way of checking gradients"""
function gradient_check(f, args...; η = 1e-5)
    g = gradient(f, args...)
    dy_expect = η*sum(abs2.(g[1]))
    dy = f(args...)-f([gi == nothing ? arg : arg.-η.*gi for (arg, gi) in zip(args, g)]...)
    @show dy
    @show dy_expect
    isapprox(dy, dy_expect, rtol=1e-2, atol=1e-8)
end

include("adjbase.jl")
include("adjYao.jl")

end # module
