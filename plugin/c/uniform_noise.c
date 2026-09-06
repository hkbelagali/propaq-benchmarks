// Uniform depolarizing-noise plugin for the propaq Noise ABI.
//
// A function of term weight alone, so it declares no dependencies and propaq collapses it
// to a single weight-indexed table before propagation starts.
//
// Ported to propaq 0.1.3's ABI: propaq_noise_damping_factor/_batch were renamed to
// propaq_noise_factor/_batch and now carry the basis kind, the term's raw key words and the
// layer position, so a plugin can read term structure or drift across the circuit. Everything
// past `weight` is unused here, and propaq_noise_depends returns 0 to say so.
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_NOISE_ABI_VERSION 1u

typedef struct { double damping; } Ctx;

uint32_t propaq_noise_abi_version(void) { return PROPAQ_NOISE_ABI_VERSION; }

uint32_t propaq_noise_depends(void) { return 0u; }

static double parse_damping(const char *json) {
    if (!json) return 0.0;
    const char *key = strstr(json, "\"damping\"");
    const char *colon = key ? strchr(key, ':') : NULL;
    return colon ? strtod(colon + 1, NULL) : 0.0;
}

void *propaq_noise_create(const char *config_json) {
    Ctx *ctx = malloc(sizeof(*ctx));
    if (ctx) ctx->damping = parse_damping(config_json);
    return ctx;
}

void propaq_noise_destroy(void *ctx) { free(ctx); }

double propaq_noise_factor(void *ctx, uint32_t basis_kind, const uint64_t *words, size_t n_words,
                           uint32_t n_units, uint32_t weight, uint32_t layer_index,
                           uint32_t n_layers) {
    (void)basis_kind; (void)words; (void)n_words; (void)n_units;
    (void)layer_index; (void)n_layers;
    return exp(-((Ctx *)ctx)->damping * (double)weight);
}

int32_t propaq_noise_factor_batch(void *ctx, uint32_t basis_kind, const uint64_t *words,
                                  size_t n_words_per_term, uint32_t n_units,
                                  const uint32_t *weights, uint32_t layer_index, uint32_t n_layers,
                                  double *out, size_t n_terms) {
    (void)basis_kind; (void)words; (void)n_words_per_term; (void)n_units;
    (void)layer_index; (void)n_layers;
    double damping = ((Ctx *)ctx)->damping;
    for (size_t i = 0; i < n_terms; ++i) out[i] = exp(-damping * (double)weights[i]);
    return 0;
}
