#!/usr/bin/env julia
# Runner: PauliPropagation.jl backend (Pauli basis only).
#
# Thread count is controlled at process start via `julia -t N` (or JULIA_NUM_THREADS), not by
# an in-script kwarg, see scaling/run_scaling_julia.sh. This script always uses the
# VectorPauliSum backend (AcceleratedKernels-parallel, auto-degrading to serial-like behavior
# at nthreads()==1) so a single code path covers both the single-threaded baseline and the
# thread-scaling sweep.
#
# Usage: julia --project=<julia_env> -t N run_pauli_propagation.jl --problem path.json
#            [--max-weight W] [--min-abs-coeff C]

using PauliPropagation
using JSON3

include(joinpath(@__DIR__, "..", "common", "circuit_ir.jl"))
using .CircuitIR

function parse_args(argv)
    d = Dict{String,String}()
    i = 1
    while i <= length(argv)
        key = argv[i]
        @assert startswith(key, "--")
        d[key[3:end]] = argv[i+1]
        i += 2
    end
    return d
end

function main()
    args = parse_args(ARGS)
    problem_path = args["problem"]
    max_weight = haskey(args, "max-weight") ? parse(Float64, args["max-weight"]) : Inf
    min_abs_coeff = haskey(args, "min-abs-coeff") ? parse(Float64, args["min-abs-coeff"]) : 1e-8

    d = CircuitIR.load_problem(problem_path)
    circuit, thetas = CircuitIR.build_circuit_and_thetas(d)
    psum0 = CircuitIR.build_observable(d)
    vpsum0 = VectorPauliSum(psum0)

    # Warm up JIT compilation on a trivial instance of the same generic function calls before
    # timing, so measured wall_time_s reflects propagation cost, not first-call compile latency.
    warmup_circ = [PauliRotation(:X, 1), CliffordGate(:H, 1)]
    warmup_psum = VectorPauliSum(PauliSum(PauliString(d.n_qubits > 0 ? d.n_qubits : 1, :Z, 1)))
    propagate(warmup_circ[1:1], warmup_psum, [0.1]; min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    propagate(warmup_circ[2:2], warmup_psum; min_abs_coeff=min_abs_coeff, max_weight=max_weight)

    t0 = time()
    result = propagate(circuit, vpsum0, thetas; min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    val = overlapwithzero(result)
    wall_time_s = time() - t0

    out = Dict(
        "backend" => "pauli_propagation_jl",
        "basis" => "pauli",
        "problem" => String(d.problem),
        "n_qubits" => d.n_qubits,
        "gate_count" => length(d.gates),
        "params" => JSON3.read(JSON3.write(d.params), Dict{String,Any}),
        "n_threads" => Threads.nthreads(),
        "wall_time_s" => wall_time_s,
        "expectation_value" => val,
        "n_terms_final" => length(result),
        "truncation_onenorm" => nothing,
        "max_terms" => nothing,
        "max_weight" => isinf(max_weight) ? nothing : max_weight,
        "min_abs_coeff" => min_abs_coeff,
    )
    println(JSON3.write(out))
end

main()
