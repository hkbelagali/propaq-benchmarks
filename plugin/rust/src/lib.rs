//! Uniform depolarizing-noise plugin for the propaq Noise ABI.
use std::ffi::{c_char, c_void, CStr};
const ABI_VERSION: u32 = 1;
struct Ctx { damping: f64 }

fn parse_damping(config: Option<&str>) -> f64 {
    let Some(text) = config else { return 0.0 };
    let Some(key) = text.find("\"damping\"") else { return 0.0 };
    let Some(colon) = text[key..].find(':') else { return 0.0 };
    text[key + colon + 1..].trim_start()
        .split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E')))
        .next().unwrap_or("").parse().unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn propaq_noise_abi_version() -> u32 { ABI_VERSION }

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_create(config_json: *const c_char) -> *mut c_void {
    let config = if config_json.is_null() { None } else { unsafe { CStr::from_ptr(config_json) }.to_str().ok() };
    Box::into_raw(Box::new(Ctx { damping: parse_damping(config) })) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_destroy(ctx: *mut c_void) {
    if !ctx.is_null() { drop(unsafe { Box::from_raw(ctx as *mut Ctx) }); }
}

/// A function of term weight alone, so propaq collapses it to one weight-indexed table.
#[no_mangle]
pub extern "C" fn propaq_noise_depends() -> u32 { 0 }

#[no_mangle]
pub extern "C" fn propaq_noise_factor(
    ctx: *mut c_void, _basis_kind: u32, _words: *const u64, _n_words: usize,
    _n_units: u32, weight: u32, _layer_index: u32, _n_layers: u32,
) -> f64 { (-unsafe { (*(ctx as *const Ctx)).damping } * f64::from(weight)).exp() }

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_factor_batch(
    ctx: *mut c_void, _basis_kind: u32, _words: *const u64, _n_words_per_term: usize,
    _n_units: u32, weights: *const u32, _layer_index: u32, _n_layers: u32,
    out: *mut f64, n_terms: usize,
) -> i32 {
    let weights = unsafe { std::slice::from_raw_parts(weights, n_terms) };
    let output = unsafe { std::slice::from_raw_parts_mut(out, n_terms) };
    let damping = unsafe { (*(ctx as *const Ctx)).damping };
    for (dst, &weight) in output.iter_mut().zip(weights) {
        *dst = (-damping * f64::from(weight)).exp();
    }
    0
}
