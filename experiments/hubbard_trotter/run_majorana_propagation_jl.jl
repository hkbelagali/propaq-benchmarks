#!/usr/bin/env julia
# Run MajoranaPropagation.jl (Majorana/fermionic basis, native side) on every saved
# hubbard_trotter native-fermionic circuit in circuits_native/.
#
# Unlike the qubit-suite backends, this package has no Qiskit bridge, so it reads the small
# parameter JSON directly with JSON3 (see ../../common/problems_fermionic.py) rather than
# through common/circuit_ir.jl's CircuitIR module (that module is for the unrelated
# qubit-suite ProblemIR shape) and builds its own circuit natively with
# hubbard_circ_fermionic_sites on an nx x ny rectangletopology, using the same t, U, dt,
# steps as the qubit-suite Hubbard problem run by the 5 qubit-side backends.
#
# This is a truly different gate-by-gate trajectory from the qubit-suite circuit (native
# fermionic gates vs Jordan-Wigner-mapped qubit gates) but describes the same physical
# model/instance size, see problems_fermionic.py's docstring.
#
# Thread count is controlled at process start via `julia -t N` (VectorMajoranaSum backend).
#
# Usage: julia --project=<julia_env> -t N experiments/hubbard_trotter/run_majorana_propagation_jl.jl

using MajoranaPropagation
using PauliPropagation: rectangletopology
using JSON3

HERE = @__DIR__
include(joinpath(HERE, "..", "..", "common", "experiment_runner.jl"))
using .ExperimentRunner

const MIN_ABS_COEFF = 1e-8

# MajoranaPropagation.jl v0.3.0 defines `coefftype` for `MajoranaSum` (src/MajoranaDataTypes.jl:162)
# but not for `VectorMajoranaSum`, even though `propagate()` (src/propagation.jl:17) calls it
# unconditionally on whichever sum type it is given. This makes VectorMajoranaSum propagation
# (needed for multithreading) crash outright as shipped. Patched here rather than upstream.
import MajoranaPropagation: coefftype
coefftype(::VectorMajoranaSum{TV,CV}) where {TV,CV} = eltype(CV)

# MajoranaPropagation.jl's `propagate()` (src/propagation.jl:20,23,29) also calls three
# PathProperties helpers (`_check_wrapping_into_paulifreqtracker`, `_checkfreqandsinfields`,
# `_check_unwrap_from_paulifreqtracker`) as bare (unqualified) names, but only imports
# PauliPropagation's exported names (`using PauliPropagation`), and these three are internal
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

function build_hubbard(params)
    nx = params["nx"]; ny = params["ny"]
    t = Float64(params["t"]); U = Float64(params["U"])
    dt = Float64(params["dt"]); steps = params["steps"]
    n_sites = nx * ny
    topology = rectangletopology(nx, ny)
    circ, thetas = hubbard_circ_fermionic_sites(topology, n_sites, steps, t, U, dt * steps)
    # Matches the qubit-suite's convention exactly (problems_qubit.py's hubbard_trotter_problem).
    # Its observable is Z on qubit n_qubits//2, but under Qiskit's little-endian Pauli-label
    # convention that list index lands on physical qubit (n_qubits-1-n_qubits//2), which for
    # n_qubits=2*n_sites simplifies to n_sites-1 (0-indexed), always spin-up on the last site,
    # for any lattice size. Z = I - 2n is the standard Jordan-Wigner convention (n=0 maps to
    # Z=+1, n=1 maps to Z=-1).
    identity_op = MajoranaSum(Float64, n_sites, Int[], true; coeff=1.0)
    n_up_last = MajoranaSum(n_sites, :nup, n_sites)  # Julia (1-indexed) site n_sites is python (0-indexed) site n_sites-1
    obs = identity_op + (-2.0) * n_up_last
    # Matches the qubit-suite's initial state exactly, checkerboard-occupied spin-up sites
    # only, spin-down empty (problems_qubit.py: `for site in range(0, n_sites, 2): qc.x(up(site))`).
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
