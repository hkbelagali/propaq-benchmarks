#!/usr/bin/env julia
# Run MajoranaPropagation.jl on the Hubbard trotter circuits.
#
# Usage: julia --project=<julia_env> -t N experiments/hubbard_trotter/run_majorana_propagation_jl.jl

using MajoranaPropagation
using PauliPropagation: rectangletopology
using JSON3

HERE = @__DIR__
using BenchCommon: ExperimentRunner

const MIN_ABS_COEFF = 1e-8

import MajoranaPropagation: coefftype
coefftype(::VectorMajoranaSum{TV,CV}) where {TV,CV} = eltype(CV)

Core.eval(MajoranaPropagation, :(_check_wrapping_into_paulifreqtracker(msum, max_freq, max_sins) = msum))
Core.eval(MajoranaPropagation, :(_check_unwrap_from_paulifreqtracker(::Type, msum) = msum))
Core.eval(MajoranaPropagation, :(function _checkfreqandsinfields(msum, max_freq, max_sins)
    if !(coefftype(msum) <: PauliPropagation.PathProperties) && ((max_freq != Inf) || (max_sins != Inf))
        throw(ArgumentError("max_freq/max_sins require PathProperties-wrapped coefficients"))
    end
end))

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

function propagate_circuit(path)
    d = JSON3.read(read(path, String))
    problem_name = String(d.problem)
    params = JSON3.read(JSON3.write(d.params), Dict{String,Any})
    if problem_name != "hubbard_trotter"
        error("unknown native fermionic problem for majorana_propagation_jl: '$problem_name'")
    end

    circ, thetas, obs0, fock, n_sites = build_hubbard(params)
    vobs0 = VectorMajoranaSum(obs0)

    result = propagate(circ, vobs0, thetas; min_abs_coeff=MIN_ABS_COEFF)
    val = overlapwithfock(result, fock)

    return Dict{String,Any}(
        "expectation_value" => val,
        "n_terms_final" => length(result),
        "truncation_onenorm" => nothing,
        "max_terms" => nothing,
        "max_weight" => nothing,
        "min_abs_coeff" => MIN_ABS_COEFF,
        "problem" => problem_name,
        "n_qubits" => n_sites,
        "gate_count" => length(circ),
        "params" => params,
    )
end

function warmup()
    warmup_circ = [MajoranaRotation(MajoranaString(2, [1, 2]))]
    warmup_obs = VectorMajoranaSum(MajoranaSum(2, :n, 1))
    propagate(warmup_circ, warmup_obs, [0.1]; min_abs_coeff=MIN_ABS_COEFF)
end

warmup()
run_on_saved_circuits(joinpath(HERE, "circuits_native"), HERE, "majorana_propagation_jl", "majorana", propagate_circuit)
