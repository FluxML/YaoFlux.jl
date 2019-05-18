using Yao, YaoBase, Zygote, LuxurySparse, YaoFlux
import Zygote: Context

t = ArrayReg(bit"0") + ArrayReg(bit"1")
normalize!(t)

# function rotmat(x)
#     I = IMatrix{2, ComplexF64}()
#     return I * cos(x/2) - im * sin(x / 2) * X
# end


# # circuit(x, y, z) = chain(Rx(x), Ry(y), Rz(z))

# # function loss(x, y, z)
# #     r = zero_state(1)
# #     # U = mat(chain(Rx(ps[1]), Rz(ps[2]), Rx(ps[3])))
# #     U = mat(chain(Rx(x), Ry(y), Rz(z)))
# #     return abs(statevec(t)' * U * statevec(r))
# # end

# # Zygote.refresh()
# # gradient(loss, 0.1, 0.1, 0.1)

# function loss(x, y, z)
#     r = zero_state(1)
#     apply!(r, Rx(0.1))
#     return abs(statevec(t)' * statevec(r))
# end

# gradient(loss, 0.1)

# (loss(0.1 + 1e-8) - loss(0.1)) / 1e-8

# Zygote.refresh()

# gradient(loss, 0.1)

function mapply!(r::ArrayReg, rb::RotationGate)
    # v0 = copy(r.state)
    # apply!(r, rb.block)
    # NOTE: we should not change register's memory address,
    # or batch operations may fail
    # r.state .= -im*sin(rb.theta/2)*r.state + cos(rb.theta/2)*r.state
    t = cos(rb.theta/2)*r.state
    copyto!(r.state, t)
    return r
end

Zygote.refresh()
_, back = Zygote._forward(x->mapply!(zero_state(1), Rx(x)), 0.1)

_, back = Zygote._forward(0.1) do x
    r = mapply!(zero_state(1), Rx(x))
    fidelity(t, r)[]
end

# _, back = Zygote._forward(r->abs(statevec(t)' * statevec(r)), zero_state(1))
# back(1)

# _, back = Zygote._forward(0.1) do x
#     r = zeros(2)
#     t = cos(x/2) * r
#     copyto!(r, t)
#     return r
# end

# st = rand(2)
# _, back = Zygote._forward(0.1) do x
#     abs(sum(x * st))
# end

# back(1)

# Zygote.refresh()
st = rand(2)
f(x) = abs(sum(x * st))

(f(0.1im+1e-8im) - f(0.1im)) / 1e-8im
f'(0.1im)
