#!/usr/bin/env julia
# Runner: MajoranaPropagation.jl backend (Majorana/fermionic basis only).
#
# Unlike the qubit-suite backends, this package has no Qiskit bridge, so it consumes a small
# *parameter* JSON (see ../common/problems_fermionic.py) and builds its own circuit natively:
#   - "hubbard_trotter": MajoranaPropagation.jl's own `hubbard_circ_fermionic_sites` builder,
#     on an nx x ny `rectangletopology`, with the same t, U, dt, steps as the qubit-suite
#     Hubbard problem run by propaq-Majorana (via Jordan-Wigner) and the 4 Pauli backends.
#   - "random_fermionic_circuit": random even-weight MajoranaRotations, following the pattern
#     in MajoranaPropagation.jl's own test/test_vector.jl `random_circuit` helper.
# These are truly different gate-by-gate trajectories from their qubit-suite counterparts
# (different RNG, native fermionic gates vs JW-mapped qubit gates) but describe the same
# physical model/instance size, see problems_fermionic.py's docstring.
#
# Thread count is controlled at process start via `julia -t N` (VectorMajoranaSum backend).
#
# Usage: julia --project=<julia_env> -t N run_majorana_propagation.jl --problem path.json
#            [--max-weight W] [--min-abs-coeff C]

using MajoranaPropagation
using PauliPropagation: rectangletopology
using JSON3
using Random

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

function build_hubbard(params)
    nx = params["nx"]; ny = params["ny"]
    t = Float64(params["t"]); U = Float64(params["U"])
    dt = Float64(params["dt"]); steps = params["steps"]
    n_sites = nx * ny
    topology = rectangletopology(nx, ny)
    circ, thetas = hubbard_circ_fermionic_sites(topology, n_sites, steps, t, U, dt * steps)
    # Matches the qubit-suite's convention exactly (problems_qubit.py's hubbard_trotter_problem).
    # Its observable is Z on qubit n_qubits//2, but under Qiskit's little-endian Pauli-label
    # convention that list index lands on *physical* qubit (n_qubits-1-n_qubits//2), which for
    # n_qubits=2*n_sites simplifies to n_sites-1 (0-indexed), always spin-up on the *last*
    # site, for any lattice size (verified against an exact Statevector cross-check on 3x3, which
    # previously measured spin-down site 0, off by both the wrong spin channel and the wrong
    # site). Z = I - 2n is the standard Jordan-Wigner convention (n=0 <-> Z=+1, n=1 <-> Z=-1).
    identity_op = MajoranaSum(Float64, n_sites, Int[], true; coeff=1.0)
    n_up_last = MajoranaSum(n_sites, :nup, n_sites)  # Julia (1-indexed) site n_sites == python (0-indexed) site n_sites-1
    obs = identity_op + (-2.0) * n_up_last
    # Matches the qubit-suite's initial state exactly: checkerboard-occupied spin-up sites only,
    # spin-down empty (problems_qubit.py: `for site in range(0, n_sites, 2): qc.x(up(site))`).
    # Previously occupied both spin channels on the checkerboard, a different filling entirely.
    fock = FockState(n_sites, 1:2:n_sites, Int[])
    return circ, thetas, obs, fock, n_sites
end

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

function main()
    args = parse_args(ARGS)
    max_weight = haskey(args, "max-weight") ? parse(Float64, args["max-weight"]) : Inf
    min_abs_coeff = haskey(args, "min-abs-coeff") ? parse(Float64, args["min-abs-coeff"]) : 1e-8

    d = JSON3.read(read(args["problem"], String))
    problem_name = String(d.problem)
    params = JSON3.read(JSON3.write(d.params), Dict{String,Any})

    if problem_name == "hubbard_trotter"
        circ, thetas, obs0, fock, n_sites = build_hubbard(params)
    elseif problem_name == "random_fermionic_circuit"
        circ, thetas, obs0, fock, n_sites = build_random_fermionic(params)
    else
        error("unknown fermionic problem '$problem_name'")
    end

    vobs0 = VectorMajoranaSum(obs0)

    # JIT warmup on a trivial instance before timing.
    warmup_circ = [MajoranaRotation(MajoranaString(n_sites, [1, 2]))]
    warmup_obs = VectorMajoranaSum(MajoranaSum(n_sites, :n, 1))
    propagate(warmup_circ, warmup_obs, [0.1]; min_abs_coeff=min_abs_coeff, max_weight=max_weight)

    t0 = time()
    result = propagate(circ, vobs0, thetas; min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    val = overlapwithfock(result, fock)
    wall_time_s = time() - t0

    out = Dict(
        "backend" => "majorana_propagation_jl",
        "basis" => "majorana",
        "problem" => problem_name,
        "n_qubits" => n_sites,
        "gate_count" => length(circ),
        "params" => params,
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
