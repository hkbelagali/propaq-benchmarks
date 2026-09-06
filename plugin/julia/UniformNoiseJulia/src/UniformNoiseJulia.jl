module UniformNoiseJulia
const PROPAQ_NOISE_ABI_VERSION = UInt32(1)

Base.@ccallable function propaq_noise_abi_version()::UInt32
    return PROPAQ_NOISE_ABI_VERSION
end

function damping_from_config(config_json::Ptr{Cchar})::Float64
    config_json == C_NULL && return 0.0
    text = unsafe_string(config_json)
    matched = match(r"\"damping\"\s*:\s*([-+0-9.eE]+)", text)
    return matched === nothing ? 0.0 : parse(Float64, matched.captures[1])
end

Base.@ccallable function propaq_noise_create(config_json::Ptr{Cchar})::Ptr{Cvoid}
    ctx = Base.Libc.malloc(sizeof(Cdouble))
    ctx == C_NULL && return C_NULL
    unsafe_store!(Ptr{Cdouble}(ctx), damping_from_config(config_json))
    return ctx
end

Base.@ccallable function propaq_noise_destroy(ctx::Ptr{Cvoid})::Cvoid
    ctx != C_NULL && Base.Libc.free(ctx)
    return
end

# A function of term weight alone, so propaq collapses it to one weight-indexed table.
Base.@ccallable function propaq_noise_depends()::UInt32
    return UInt32(0)
end

Base.@ccallable function propaq_noise_factor(
    ctx::Ptr{Cvoid}, basis_kind::UInt32, words::Ptr{UInt64}, n_words::Csize_t,
    n_units::UInt32, weight::UInt32, layer_index::UInt32, n_layers::UInt32,
)::Cdouble
    damping = unsafe_load(Ptr{Cdouble}(ctx))
    return exp(-damping * Float64(weight))
end

Base.@ccallable function propaq_noise_factor_batch(
    ctx::Ptr{Cvoid}, basis_kind::UInt32, words::Ptr{UInt64}, n_words_per_term::Csize_t,
    n_units::UInt32, weights::Ptr{UInt32}, layer_index::UInt32, n_layers::UInt32,
    out::Ptr{Cdouble}, n_terms::Csize_t,
)::Int32
    weight_view = unsafe_wrap(Array, weights, n_terms)
    output = unsafe_wrap(Array, out, n_terms)
    damping = unsafe_load(Ptr{Cdouble}(ctx))
    @inbounds for i in eachindex(output)
        output[i] = exp(-damping * Float64(weight_view[i]))
    end
    return Int32(0)
end
end
