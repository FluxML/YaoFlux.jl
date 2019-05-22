# generate a Heisenberg Model Hamiltonian
using Yao
using YaoFlux
using LinearAlgebra
using QuAlgorithmZoo: heisenberg, random_diff_circuit, pair_ring

nbit = 6
h = mat(heisenberg(nbit))
v0 = statevec(zero_state(nbit))
function energy(circuit)
    v = mat(circuit) * v0
    (v'* h * v)[] |> real
end

# a circuit as ansatz
#circuit = chain(nbit, [put(2, 1=>H), put(2, 2=>H), put(2, 2=>Rz(0.0)), put(2, 2=>Rx(0.0)), control(2, 1, 2=>Z), put(2, 2=>Rx(0.0)), put(2, 2=>Rz(0.0))])
circuit = random_diff_circuit(nbit, 2, pair_ring(nbit))

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
EG = eigvals(Matrix(h))[1]
println("Exact Energy is $EG")
train!(energy, circuit, ADAM(0.1); maxiter=200)

