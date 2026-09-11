# All the experiments call this module to run for the saved circuits
module ExperimentRunner

using JSON3

export run_on_saved_circuits

function _natural_sort_key(path::AbstractString)
    stem = splitext(basename(path))[1]
    return replace(stem, r"\d+" => (m -> lpad(m, 10, '0')))
end

function circuit_paths(circuits_dir::AbstractString)
    paths = filter(p -> endswith(p, ".json"), readdir(circuits_dir; join=true))
    sort(paths; by=_natural_sort_key)
end

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
