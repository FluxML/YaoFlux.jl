using Yao, YaoBase, Zygote, LuxurySparse, YaoFlux
import Zygote: Context
import YaoBase.Const: X

t = ArrayReg(bit"0") + ArrayReg(bit"1")
normalize!(t)

function rotmat(x)
    I = IMatrix{2, ComplexF64}()
    return I * cos(x/2) - im * sin(x / 2) * X
end


circuit(x, y, z) = chain(Rx(x), Ry(y), Rz(z))

function loss(x, y, z)
    r = zero_state(1)
    # U = mat(chain(Rx(ps[1]), Rz(ps[2]), Rx(ps[3])))
    U = mat(chain(Rx(x), Ry(y), Rz(z)))
    return abs(statevec(t)' * U * statevec(r))
end

Zygote.refresh()
gradient(loss, 0.1, 0.1, 0.1)
