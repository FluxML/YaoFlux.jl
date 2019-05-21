# generate a Heisenberg Model Hamiltonian
using QuAlgorithmZoo: heisenberg

h = mat(heisenberg(2))
v0 = statevec(zero_state(2))
function energy(circuit)
    v = mat(circuit) * v0
    (v'* h * v)[] |> real
end

# a circuit as ansatz
circuit = chain(2, [put(2, 1=>H), put(2, 2=>H), put(2, 2=>Rz(0.0)), put(2, 2=>Rx(0.0)), control(2, 1, 2=>Z), put(2, 2=>Rx(0.0)), put(2, 2=>Rz(0.0))])

# obtain the energy
energy(circuit)

# get the gradient with respect to circuit.
gst = collect_gradients(energy'(circuit))

using Flux: ADAM, Optimise
function train!(lossfunc, circuit, optimizer; maxiter::Int=200)
    dispatch!(circuit, :random)
    params = parameters(circuit)
    for i = 1:maxiter
        grad = collect_gradients(lossfunc'(circuit))
        Optimise.update!(optimizer, params, grad)
        dispatch!(circuit, params)
        println("Iter $i, Loss = $(lossfunc(circuit))")
    end
    circuit
end

using Random
Random.seed!(5)
train!(ec4, circuit, ADAM(0.1); maxiter=100)
