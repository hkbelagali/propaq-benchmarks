# Shared helper that every experiment's Julia backend runner file calls.
#
# Mirrors propaq_benchmarks/experiment_runner.py. There is no Python orchestrator anymore, so this
# writes only a JSONL checkpoint (results_<backend>.jsonl) directly next to the experiment,
# not an npz snapshot (Julia has no numpy-compatible npz writer in this environment).
# Every plotting script reads both a backend's .npz (if present) and its .jsonl (if not)
# through propaq_benchmarks/io_utils.py's load_experiment_results, so this is not a gap, just a
# different file for the same row data.
module ExperimentRunner

using JSON3

export run_on_saved_circuits

function _natural_sort_key(path::AbstractString)
    # Zero-pads every run of digits to a fixed width so plain string comparison already
    # puts steps2 before steps10. Plain alphabetical sort on the unpadded name puts steps10
    # before steps2, which processes a fine step curve out of numeric order and makes a
    # partial run's progress confusing to read.
    stem = splitext(basename(path))[1]
    return replace(stem, r"\d+" => (m -> lpad(m, 10, '0')))
end

function circuit_paths(circuits_dir::AbstractString)
    paths = filter(p -> endswith(p, ".json"), readdir(circuits_dir; join=true))
    sort(paths; by=_natural_sort_key)
end

"""
    run_on_saved_circuits(circuits_dir, experiment_dir, backend, basis, propagate)

Call `propagate(path)` once per saved circuit JSON file in `circuits_dir`. `propagate`
must return a `Dict` of result fields including at least `n_qubits`, `problem`, and
`params`, plus whatever the backend measured. `wall_time_s` is filled in automatically
around the call unless `propagate` already set it. A circuit whose label is already
recorded with `ok=true` in the existing checkpoint is skipped.
"""
function run_on_saved_circuits(circuits_dir, experiment_dir, backend::AbstractString,
                                basis::AbstractString, propagate::Function)
    checkpoint = joinpath(experiment_dir, "results_$(backend).jsonl")
    done = Set{String}()
    if isfile(checkpoint)
        for line in eachline(checkpoint)
            isempty(strip(line)) && continue
            rec = JSON3.read(line)
            if get(rec, :ok, false) && get(rec, :basis, "") == basis
                push!(done, String(rec.label))
            end
        end
    end

    paths = circuit_paths(circuits_dir)
    isempty(paths) && error("no saved circuits in $circuits_dir, run generate_circuits.py first")

    for path in paths
        label = splitext(basename(path))[1]
        if label in done
            println("  $label ... skip (already done)")
            continue
        end
        t0 = time()
        local record
        try
            record = propagate(path)
            record["ok"] = true
        catch e
            record = Dict{String,Any}("ok" => false, "error" => sprint(showerror, e))
        end
        if !haskey(record, "wall_time_s")
            record["wall_time_s"] = time() - t0
        end
        record["label"] = label
        record["backend"] = backend
        record["basis"] = basis
        record["n_threads"] = Threads.nthreads()
        status = record["ok"] ? "OK" : "FAIL"
        println("  $label ... $status ($(round(record["wall_time_s"], digits=3))s)")
        record["ok"] || println("    $(record["error"])")
        open(checkpoint, "a") do io
            println(io, JSON3.write(record))
        end
    end
end

end
