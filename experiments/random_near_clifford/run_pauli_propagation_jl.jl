#!/usr/bin/env julia
# Run PauliPropagation.jl (Pauli basis) on every saved random_near_clifford circuit.
#
# Thread count is controlled at process start via `julia -t N`, not by an in-script kwarg.
# Always uses the VectorPauliSum backend (AcceleratedKernels-parallel, auto-degrading to
# serial-like behavior at nthreads()==1), so a single code path covers 1 thread and many.
#
# Usage: julia --project=<julia_env> -t 64 experiments/random_near_clifford/run_pauli_propagation_jl.jl

using PauliPropagation
using JSON3

HERE = @__DIR__
include(joinpath(HERE, "..", "..", "common", "circuit_ir.jl"))
include(joinpath(HERE, "..", "..", "common", "experiment_runner.jl"))
using .CircuitIR
using .ExperimentRunner

const MIN_ABS_COEFF = 1e-6

function propagate_circuit(path)
    d = CircuitIR.load_problem(path)
    circuit, thetas = CircuitIR.build_circuit_and_thetas(d)
    psum0 = CircuitIR.build_observable(d)
    vpsum0 = VectorPauliSum(psum0)

    result = propagate(circuit, vpsum0, thetas; min_abs_coeff=MIN_ABS_COEFF)
    val = overlapwithzero(result)

    return Dict{String,Any}(
        "expectation_value" => val,
        "n_terms_final" => length(result),
        "min_abs_coeff" => MIN_ABS_COEFF,
        "n_threads" => Threads.nthreads(),
        "problem" => String(d.problem),
        "n_qubits" => d.n_qubits,
        "gate_count" => length(d.gates),
        "params" => JSON3.read(JSON3.write(d.params), Dict{String,Any}),
    )
end

function warmup()
    warmup_circ = [PauliRotation(:X, 1), CliffordGate(:H, 1)]
    warmup_psum = VectorPauliSum(PauliSum(PauliString(1, :Z, 1)))
    propagate(warmup_circ[1:1], warmup_psum, [0.1]; min_abs_coeff=MIN_ABS_COEFF)
    propagate(warmup_circ[2:2], warmup_psum; min_abs_coeff=MIN_ABS_COEFF)
end

warmup()
run_on_saved_circuits(joinpath(HERE, "circuits"), HERE, "pauli_propagation_jl", "pauli", propagate_circuit)
