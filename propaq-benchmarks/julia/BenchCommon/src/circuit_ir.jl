"""
Julia side parser for the circuits
"""
module CircuitIR

using JSON3
using PauliPropagation

export load_problem, build_circuit_and_thetas, build_observable

function load_problem(path::String)
    return JSON3.read(read(path, String))
end

function build_circuit_and_thetas(d)
    n = d.n_qubits
    circuit = PauliPropagation.Gate[]
    thetas = Float64[]
    for g in d.gates
        name = g.name
        qs = [q + 1 for q in g.qubits]
        if name == "rz"
            push!(circuit, PauliRotation(:Z, qs[1]))
            push!(thetas, g.angle)
        elseif name == "rx"
            push!(circuit, PauliRotation(:X, qs[1]))
            push!(thetas, g.angle)
        elseif name == "ry"
            push!(circuit, PauliRotation(:Y, qs[1]))
            push!(thetas, g.angle)
        elseif name == "rzz"
            push!(circuit, PauliRotation([:Z, :Z], qs))
            push!(thetas, g.angle)
        elseif name == "cx"
            push!(circuit, CliffordGate(:CNOT, qs))
        elseif name == "h"
            push!(circuit, CliffordGate(:H, qs))
        else
            error("gate '$name' outside COMMON_BASIS")
        end
    end
    return circuit, thetas
end

function build_observable(d)
    n = d.n_qubits
    psum = PauliSum(n)
    for (label, coeff) in zip(d.observable.paulis, d.observable.coeffs)
        s = String(label)
        symbols = Symbol[]
        qinds = Int[]
        for (k, ch) in enumerate(s)
            if ch != 'I'
                push!(symbols, Symbol(string(ch)))
                push!(qinds, n - k + 1)
            end
        end
        isempty(symbols) && error("identity observable term not supported by this IR reader")
        pstr = PauliString(n, symbols, qinds, Float64(coeff))
        add!(psum, PauliSum(pstr))
    end
    return psum
end

end # module
