using Yao, Zygote, YaoFlux, YaoArrayRegister, YaoBlocks
import Yao.YaoBlocks: RotationGate

# a, back = Zygote._forward(apply!, rand_state(1), Rx(0.1))

# back(rand_state(1))

Zygote.refresh()

r, back = Zygote._forward(0.1, 0.1) do x, y
    return apply!(zero_state(1), chain(Rx(x), Rz(y)))
end

back(YaoFlux.GradReg(r, rand_state(1)))


_, back = Zygote._forward(fidelity, rand_state(1), rand_state(1))
back([1])


t = rand_state(1)
r = zero_state(1)
gradient(x->fidelity(t, apply!(r, Rx(x)))[], 0.1)

Zygote._forward(Rx, 0.1)

_, back = Zygote._forward((x, y)->abs(statevec(x)' * statevec(y)), rand_state(1), rand_state(1))
back(1)


_, back = Zygote._forward(state, rand_state(1))
back(rand(2))
