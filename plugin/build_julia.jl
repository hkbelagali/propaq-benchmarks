using PackageCompiler
root = @__DIR__
create_library(joinpath(root, "julia", "UniformNoiseJulia"), joinpath(root, "build", "julia");
    # Keep all stdlibs and their binary artifacts (notably OpenBLAS). A plugin
    # runs inside Python, so it cannot rely on Julia's original installation.
    lib_name="uniform_noise_julia", force=true, filter_stdlibs=false, incremental=true)
