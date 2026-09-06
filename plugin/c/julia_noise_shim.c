// Runs a PackageCompiler Julia plugin on one dedicated Julia-owned thread.
// propaq invokes plugins from arbitrary Rayon workers, and those foreign threads
// cannot directly enter Julia because its GC state is thread-local.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>

typedef void (*init_julia_fn)(int, char **);
typedef uint32_t (*abi_version_fn)(void);
typedef void *(*create_fn)(const char *);
typedef void (*destroy_fn)(void *);
// propaq 0.1.3's noise ABI: the factor callbacks additionally carry the basis kind, the
// term's raw key words and the layer position. This plugin reads none of them (it is a
// function of weight alone) but the shim must still forward them, since it stands in for
// the Julia library at the ABI boundary and the argument lists have to match exactly.
typedef double (*factor_fn)(void *, uint32_t, const uint64_t *, size_t, uint32_t, uint32_t,
                            uint32_t, uint32_t);
typedef int32_t (*batch_fn)(void *, uint32_t, const uint64_t *, size_t, uint32_t,
                            const uint32_t *, uint32_t, uint32_t, double *, size_t);

enum request_kind { CREATE, DESTROY, FACTOR, BATCH };
struct request {
    enum request_kind kind;
    const char *config;
    void *ctx;
    uint32_t basis_kind, n_units, weight, layer_index, n_layers;
    const uint64_t *words;
    size_t n_words;
    const uint32_t *weights;
    double *out, factor;
    size_t n;
    int32_t rc;
    void *created;
};

static pthread_once_t startup_once = PTHREAD_ONCE_INIT;
static pthread_t julia_thread;
static pthread_mutex_t request_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t call_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t request_ready = PTHREAD_COND_INITIALIZER;
static pthread_cond_t request_done = PTHREAD_COND_INITIALIZER;
static int initialized, pending, completed;
static uint32_t abi_version_value;
static struct request request;
static create_fn create_noise;
static destroy_fn destroy_noise;
static factor_fn noise_factor;
static batch_fn noise_factor_batch;

static void *julia_host(void *unused) {
    (void)unused;
    void *plugin = dlopen("libuniform_noise_julia.so", RTLD_NOW | RTLD_GLOBAL);
    init_julia_fn init_julia = plugin ? (init_julia_fn)dlsym(plugin, "init_julia") : NULL;
    if (init_julia) {
        char *argv[] = {"propaq-julia-plugin", NULL};
        init_julia(1, argv);
        abi_version_fn abi = (abi_version_fn)dlsym(plugin, "propaq_noise_abi_version");
        create_noise = (create_fn)dlsym(plugin, "propaq_noise_create");
        destroy_noise = (destroy_fn)dlsym(plugin, "propaq_noise_destroy");
        noise_factor = (factor_fn)dlsym(plugin, "propaq_noise_factor");
        noise_factor_batch = (batch_fn)dlsym(plugin, "propaq_noise_factor_batch");
        abi_version_value = abi ? abi() : 0;
    }
    pthread_mutex_lock(&request_mutex);
    initialized = 1;
    pthread_cond_broadcast(&request_done);
    for (;;) {
        while (!pending) pthread_cond_wait(&request_ready, &request_mutex);
        struct request *r = &request;
        pthread_mutex_unlock(&request_mutex);
        if (r->kind == CREATE) r->created = create_noise ? create_noise(r->config) : NULL;
        else if (r->kind == DESTROY && destroy_noise) destroy_noise(r->ctx);
        else if (r->kind == FACTOR)
            r->factor = noise_factor ? noise_factor(r->ctx, r->basis_kind, r->words, r->n_words,
                                                    r->n_units, r->weight, r->layer_index,
                                                    r->n_layers) : 0.0;
        else if (r->kind == BATCH)
            r->rc = noise_factor_batch ? noise_factor_batch(r->ctx, r->basis_kind, r->words,
                                                            r->n_words, r->n_units, r->weights,
                                                            r->layer_index, r->n_layers, r->out,
                                                            r->n) : -1;
        pthread_mutex_lock(&request_mutex);
        pending = 0;
        completed = 1;
        pthread_cond_broadcast(&request_done);
    }
    return NULL;
}

static void start_julia(void) {
    pthread_create(&julia_thread, NULL, julia_host, NULL);
    pthread_mutex_lock(&request_mutex);
    while (!initialized) pthread_cond_wait(&request_done, &request_mutex);
    pthread_mutex_unlock(&request_mutex);
}

static int begin_request(enum request_kind kind) {
    pthread_once(&startup_once, start_julia);
    if (abi_version_value != 1) return 0;
    pthread_mutex_lock(&call_mutex);
    pthread_mutex_lock(&request_mutex);
    request.kind = kind;
    completed = 0;
    return 1;
}

static void dispatch_and_finish(void) {
    pending = 1;
    pthread_cond_signal(&request_ready);
    while (!completed) pthread_cond_wait(&request_done, &request_mutex);
    pthread_mutex_unlock(&request_mutex);
    pthread_mutex_unlock(&call_mutex);
}

uint32_t propaq_noise_abi_version(void) {
    pthread_once(&startup_once, start_julia);
    return abi_version_value;
}

void *propaq_noise_create(const char *config) {
    if (!begin_request(CREATE)) return NULL;
    request.config = config;
    dispatch_and_finish();
    return request.created;
}

void propaq_noise_destroy(void *ctx) {
    if (!begin_request(DESTROY)) return;
    request.ctx = ctx;
    dispatch_and_finish();
}

// Weight alone, matching the Julia plugin it fronts, so propaq tabulates it once.
uint32_t propaq_noise_depends(void) { return 0u; }

double propaq_noise_factor(void *ctx, uint32_t basis_kind, const uint64_t *words, size_t n_words,
                           uint32_t n_units, uint32_t weight, uint32_t layer_index,
                           uint32_t n_layers) {
    if (!begin_request(FACTOR)) return 0.0;
    request.ctx = ctx; request.basis_kind = basis_kind; request.words = words;
    request.n_words = n_words; request.n_units = n_units; request.weight = weight;
    request.layer_index = layer_index; request.n_layers = n_layers;
    dispatch_and_finish();
    return request.factor;
}

int32_t propaq_noise_factor_batch(void *ctx, uint32_t basis_kind, const uint64_t *words,
                                  size_t n_words_per_term, uint32_t n_units,
                                  const uint32_t *weights, uint32_t layer_index, uint32_t n_layers,
                                  double *out, size_t n_terms) {
    if (!begin_request(BATCH)) return -1;
    request.ctx = ctx; request.basis_kind = basis_kind; request.words = words;
    request.n_words = n_words_per_term; request.n_units = n_units; request.weights = weights;
    request.layer_index = layer_index; request.n_layers = n_layers;
    request.out = out; request.n = n_terms;
    dispatch_and_finish();
    return request.rc;
}
