#!/usr/bin/env julia
# Run PauliPropagation.jl on the random_circuit instance, at whatever thread count this
# process was launched with (julia -t N).
#
# Unlike the Python runners in this folder, Julia's thread count is fixed for the whole
# process at launch, so this file records one row per invocation instead of sweeping
# THREAD_SWEEP itself. Build the thread-scaling curve by running this file once per thread
# count you want a point for, e.g.:
#   julia --project=<julia_env> -t 1  experiments/thread_scaling/run_pauli_propagation_jl.jl
#   julia --project=<julia_env> -t 8  experiments/thread_scaling/run_pauli_propagation_jl.jl
#   julia --project=<julia_env> -t 32 experiments/thread_scaling/run_pauli_propagation_jl.jl
# Each run appends its own row, keyed by n_threads, so repeats at the same thread count are
# skipped and different thread counts accumulate into the same results_pauli_propagation_jl.jsonl.

using PauliPropagation
using JSON3

HERE = @__DIR__
include(joinpath(HERE, "..", "..", "common", "circuit_ir.jl"))
using .CircuitIR

const MIN_ABS_COEFF = 1e-6

function main()
    checkpoint = joinpath(HERE, "results_pauli_propagation_jl.jsonl")
    n_threads = Threads.nthreads()
    if isfile(checkpoint)
        for line in eachline(checkpoint)
            isempty(strip(line)) && continue
            rec = JSON3.read(line)
            if get(rec, :ok, false) && rec.label == "random_circuit" && rec.n_threads == n_threads
                println("  random_circuit n_threads=$n_threads ... skip (already done)")
                return
            end
        end
    end

    d = CircuitIR.load_problem(joinpath(HERE, "circuits", "random_circuit.json"))
    circuit, thetas = CircuitIR.build_circuit_and_thetas(d)
    psum0 = CircuitIR.build_observable(d)
    vpsum0 = VectorPauliSum(psum0)

    warmup_circ = [PauliRotation(:X, 1), CliffordGate(:H, 1)]
    warmup_psum = VectorPauliSum(PauliSum(PauliString(d.n_qubits, :Z, 1)))
    propagate(warmup_circ[1:1], warmup_psum, [0.1]; min_abs_coeff=MIN_ABS_COEFF)
    propagate(warmup_circ[2:2], warmup_psum; min_abs_coeff=MIN_ABS_COEFF)

    t0 = time()
    result = propagate(circuit, vpsum0, thetas; min_abs_coeff=MIN_ABS_COEFF)
    val = overlapwithzero(result)
    wall_time_s = time() - t0

    record = Dict{String,Any}(
        "ok" => true,
        "label" => "random_circuit",
        "basis" => "pauli",
        "backend" => "pauli_propagation_jl",
        "n_threads" => n_threads,
        "wall_time_s" => wall_time_s,
        "expectation_value" => val,
        "n_terms_final" => length(result),
        "problem" => String(d.problem),
        "n_qubits" => d.n_qubits,
    )
    println("  random_circuit n_threads=$n_threads ... OK ($(round(wall_time_s, digits=3))s)")
    open(checkpoint, "a") do io
        println(io, JSON3.write(record))
    end
end

main()
