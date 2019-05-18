using Yao, YaoFlux, Zygote

r = ComplexF64[1, 0]
t = rand(ComplexF64, 2)

function f1!(r, x)
    tt = -im * sin(x) * r

    for k in 1:length(tt)
        r[k] = tt[k]
    end
    # copyto!(r, tt)
    return abs(t' * r)
end

function f2!(r, x)
    tt = -im * sin(x) * r
    return abs(t' * tt)
end

Zygote.refresh()
gradient(f1!, ComplexF64[1, 0], 0.1)[2]

δ = sqrt(eps())
(f1!(ComplexF64[1, 0], 0.1+δ/2) - f1!(ComplexF64[1, 0], 0.1-δ/2)) / δ

gradient(f2!, ComplexF64[1, 0], 0.1)[2]
(f2!(ComplexF64[1, 0], 0.1+δ/2) - f2!(ComplexF64[1, 0], 0.1-δ/2)) / δ
