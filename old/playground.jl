using Zygote: @adjoint!, grad_mut, _forward
function _forward(cx::Context, ::typeof(collect), g::Base.Generator)

@adjoint! function dispatch!(circuit, params)
    dispatch!(circuit, params),
    function (x_)
        dstk = grad_mut(__context__, circuit)
        @show dstk
        dstk = grad_mut(__context__, params)
        @show dstk
        @show x_
        (nothing, grad)
    end
end

gradient_check(l1, [0.3, 0.6])

v = randn(ComplexF64, 4)
reg = rand_state(4)

cnot_mat = mat(control(2, 1, 2=>X))
function loss4(params::Vector)
    dispatch!(circuit, params)
    M = mat(circuit)
    norm(M-cnot_mat)
end

@adjoint invperm(x) = invperm(x), adjy -> nothing
@adjoint function norm(x)
    y = norm(x)
    y, adjy->(adjy/y*x,)
end

gradient_check(norm, randn(ComplexF64, 3,3))

loss4([0.3, 0.1, 0.3, 0.5])
loss4'([0.3, 0.2, 0.3, 0.5])

norm
function l2(A)
    norm(A - B)
end
