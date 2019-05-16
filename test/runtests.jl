using YaoFlux
using Test, LuxurySparse, Zygote, YaoBase

f(x) = real(sum(x * Const.X))
f'(0.1)

f'(0.1)