#!/usr/bin/env julia
# Run MajoranaPropagation.jl (Majorana/fermionic basis only) on every saved native-fermionic
# random_fermionic_circuit instance in circuits_native/.
#
# Unlike the qubit-suite backends, this package has no Qiskit bridge, so it consumes the small
# *parameter* JSON written by propaq_benchmarks/problems_fermionic.py's random_fermionic_circuit_fermionic
# (not the ProblemIR shape in circuits/, which is for the qubit-suite backends only) and builds
# its own circuit natively from random even-weight MajoranaRotations, following the pattern in
# MajoranaPropagation.jl's own test/test_vector.jl `random_circuit` helper. This is a truly
# different gate-by-gate trajectory from the qubit-suite circuit of the same nominal size
# (different RNG, native fermionic gates vs JW-mapped qubit gates) but describes the same
# physical instance size, see propaq_benchmarks/problems_fermionic.py's module docstring.
#
# Thread count is controlled at process start via `julia -t N` (VectorMajoranaSum backend).
#
# Usage: julia --project=<julia_env> -t 64 experiments/random_fermionic_circuit/run_majorana_propagation_jl.jl

using MajoranaPropagation
using JSON3
using Random

HERE = @__DIR__
using BenchCommon: ExperimentRunner

# MajoranaPropagation.jl v0.3.0 defines `coefftype` for `MajoranaSum` (src/MajoranaDataTypes.jl:162)
# but not for `VectorMajoranaSum`, even though `propagate()` (src/propagation.jl:17) calls it
# unconditionally on whichever sum type it's given. This makes VectorMajoranaSum propagation
# (needed for multithreading) crash outright as shipped. Patched here rather than upstream.
import MajoranaPropagation: coefftype
coefftype(::VectorMajoranaSum{TV,CV}) where {TV,CV} = eltype(CV)

# MajoranaPropagation.jl's `propagate()` (src/propagation.jl:20,23,29) also calls three
# PathProperties helpers (`_check_wrapping_into_paulifreqtracker`, `_checkfreqandsinfields`,
# `_check_unwrap_from_paulifreqtracker`) as bare (unqualified) names, but only imports
# PauliPropagation's *exported* names (`using PauliPropagation`), and these three are internal
# (underscore-prefixed, non-exported), so every `propagate()` call on any AbstractMajoranaSum
# crashes with UndefVarError as shipped. We never pass max_freq/max_sins (both default Inf),
# so only the generic no-op fallback branches of these three functions are ever reachable here.
# They are injected verbatim from PauliPropagation.PathProperties's own generic (non-PauliSum)
# fallback implementations.
Core.eval(MajoranaPropagation, :(_check_wrapping_into_paulifreqtracker(msum, max_freq, max_sins) = msum))
Core.eval(MajoranaPropagation, :(_check_unwrap_from_paulifreqtracker(::Type, msum) = msum))
Core.eval(MajoranaPropagation, :(function _checkfreqandsinfields(msum, max_freq, max_sins)
    if !(coefftype(msum) <: PauliPropagation.PathProperties) && ((max_freq != Inf) || (max_sins != Inf))
        throw(ArgumentError("max_freq/max_sins require PathProperties-wrapped coefficients"))
    end
end))

const MIN_ABS_COEFF = 1e-6

function build_random_fermionic(params)
    nfermions = params["n_modes"]
    n_gates = params["n_gates"]
    seed = params["seed"]
    maxW = get(params, "max_weight_per_gate", 4)
    Random.seed!(seed)

    circ = MajoranaRotation[]
    thetas = Float64[]
    for _ = 1:n_gates
        ms_weight = rand(1:(maxW ÷ 2)) * 2
        gammas = Int[]
        while length(gammas) < ms_weight
            g = rand(1:2*nfermions)
            g ∉ gammas && push!(gammas, g)
        end
        push!(circ, MajoranaRotation(MajoranaString(nfermions, gammas)))
        push!(thetas, rand() * 2 * pi)
    end
    obs = MajoranaSum(nfermions, :n, max(1, nfermions ÷ 2))
    fock = FockState(nfermions, 1:2:nfermions)
    return circ, thetas, obs, fock, nfermions
end

function propagate_circuit(path)
    d = JSON3.read(read(path, String))
    problem_name = String(d.problem)
    @assert problem_name == "random_fermionic_circuit" "unexpected problem '$problem_name' in circuits_native/"
    params = JSON3.read(JSON3.write(d.params), Dict{String,Any})

    circ, thetas, obs0, fock, n_modes = build_random_fermionic(params)
    vobs0 = VectorMajoranaSum(obs0)

    result = propagate(circ, vobs0, thetas; min_abs_coeff=MIN_ABS_COEFF)
    val = overlapwithfock(result, fock)

    return Dict{String,Any}(
        "expectation_value" => val,
        "n_terms_final" => length(result),
        "min_abs_coeff" => MIN_ABS_COEFF,
        "problem" => problem_name,
        "n_qubits" => n_modes,
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
