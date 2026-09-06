#!/usr/bin/env julia
# Run MajoranaPropagation.jl on the native hubbard instance, at whatever thread count this
# process was launched with (julia -t N). See run_pauli_propagation_jl.jl in this same
# folder for why this records one row per invocation instead of sweeping in-process, and
# build the thread-scaling curve the same way, by running this file once per thread count.

using MajoranaPropagation
using PauliPropagation: rectangletopology
using JSON3

# MajoranaPropagation.jl v0.3.0 defines coefftype for MajoranaSum but not VectorMajoranaSum,
# even though propagate() calls it unconditionally on whichever sum type it is given, which
# makes VectorMajoranaSum propagation (needed for multithreading) crash outright as shipped.
import MajoranaPropagation: coefftype
coefftype(::VectorMajoranaSum{TV,CV}) where {TV,CV} = eltype(CV)

# propagate() also calls three PathProperties helpers as bare names without importing them,
# which crashes every call with UndefVarError as shipped. Only the generic no-op fallback
# branches of these three functions are ever reachable here, since max_freq/max_sins are
# never passed (both default Inf), so they are injected verbatim from PauliPropagation's
# own generic fallback implementations.
Core.eval(MajoranaPropagation, :(_check_wrapping_into_paulifreqtracker(msum, max_freq, max_sins) = msum))
Core.eval(MajoranaPropagation, :(_check_unwrap_from_paulifreqtracker(::Type, msum) = msum))
Core.eval(MajoranaPropagation, :(function _checkfreqandsinfields(msum, max_freq, max_sins)
    if !(coefftype(msum) <: PauliPropagation.PathProperties) && ((max_freq != Inf) || (max_sins != Inf))
        throw(ArgumentError("max_freq/max_sins require PathProperties-wrapped coefficients"))
    end
end))

HERE = @__DIR__
const MIN_ABS_COEFF = 1e-6

function build_hubbard(params)
    nx = params["nx"]; ny = params["ny"]
    t = Float64(params["t"]); U = Float64(params["U"])
    dt = Float64(params["dt"]); steps = params["steps"]
    n_sites = nx * ny
    topology = rectangletopology(nx, ny)
    circ, thetas = hubbard_circ_fermionic_sites(topology, n_sites, steps, t, U, dt * steps)
    identity_op = MajoranaSum(Float64, n_sites, Int[], true; coeff=1.0)
    n_up_last = MajoranaSum(n_sites, :nup, n_sites)
    obs = identity_op + (-2.0) * n_up_last
    fock = FockState(n_sites, 1:2:n_sites, Int[])
    return circ, thetas, obs, fock, n_sites
end

function main()
    checkpoint = joinpath(HERE, "results_majorana_propagation_jl.jsonl")
    n_threads = Threads.nthreads()
    if isfile(checkpoint)
        for line in eachline(checkpoint)
            isempty(strip(line)) && continue
            rec = JSON3.read(line)
            if get(rec, :ok, false) && rec.label == "hubbard_native" && rec.n_threads == n_threads
                println("  hubbard_native n_threads=$n_threads ... skip (already done)")
                return
            end
        end
    end

    d = JSON3.read(read(joinpath(HERE, "circuits_native", "hubbard_native.json"), String))
    params = JSON3.read(JSON3.write(d.params), Dict{String,Any})
    circ, thetas, obs0, fock, n_sites = build_hubbard(params)
    vobs0 = VectorMajoranaSum(obs0)

    warmup_circ = [MajoranaRotation(MajoranaString(n_sites, [1, 2]))]
    warmup_obs = VectorMajoranaSum(MajoranaSum(n_sites, :n, 1))
    propagate(warmup_circ, warmup_obs, [0.1]; min_abs_coeff=MIN_ABS_COEFF)

    t0 = time()
    result = propagate(circ, vobs0, thetas; min_abs_coeff=MIN_ABS_COEFF)
    val = overlapwithfock(result, fock)
    wall_time_s = time() - t0

    record = Dict{String,Any}(
        "ok" => true,
        "label" => "hubbard_native",
        "basis" => "majorana",
        "backend" => "majorana_propagation_jl",
        "n_threads" => n_threads,
        "wall_time_s" => wall_time_s,
        "expectation_value" => val,
        "n_terms_final" => length(result),
        "problem" => String(d.problem),
        "n_qubits" => n_sites,
    )
    println("  hubbard_native n_threads=$n_threads ... OK ($(round(wall_time_s, digits=3))s)")
    open(checkpoint, "a") do io
        println(io, JSON3.write(record))
    end
end

main()
