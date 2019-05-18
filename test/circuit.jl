using Yao, Zygote, YaoFlux
import Zygote: Context


function mapply!(r::ArrayReg, rb::RotationGate)
    # v0 = copy(r.state)
    # apply!(r, rb.block)
    # NOTE: we should not change register's memory address,
    # or batch operations may fail
    # r.state .= -im*sin(rb.theta/2)*r.state + cos(rb.theta/2)*r.state
    t = -im*sin(rb.theta/2)*r.state
    copyto!(r.state, t)
    return r
end

t = ArrayReg(bit"0") + ArrayReg(bit"1")
normalize!(t)

function loss(x)
    r = mapply!(zero_state(1), Rx(x))
    return fidelity(t, r)[]    
end

Zygote.refresh()
_, back = Zygote._forward(loss, 0.1)

back(1)
(loss(0.1+1e-8) - loss(0.1)) / 1e-8

r = ComplexF64[1, 0]
t = rand(ComplexF64, 2)

function f1!(r, x)
    tt = -im * sin(x) * r
    copyto!(r, tt)
    return abs(t' * r)
end

function f2!(r, x)
    tt = -im * sin(x) * r
    return abs(t' * r)
end

Zygote.refresh()
gradient(f1!, r, 0.1)[2]

(f1!(statevec(zero_state(1)), 0.1+sqrt(eps())/2) - f1!(statevec(zero_state(1)), 0.1-sqrt(eps())/2)) / sqrt(eps())

gradient(f2!, r, 0.1)[2]
(f2!(statevec(zero_state(1)), 0.1+1e-8) - f2!(statevec(zero_state(1)), 0.1)) / 1e-8
