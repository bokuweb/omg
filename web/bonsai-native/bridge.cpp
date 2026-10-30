// Minimal CPU-only Prism llama.cpp bridge. Model files are Blob-backed WORKERFS
// files, so loading does not retain a second 950 MB copy in the WASM heap.
#include "llama.h"
#include <emscripten.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static llama_model * model = nullptr;
static llama_context * ctx = nullptr;
static std::string error;
static std::vector<llama_token> tokens, resident;
static std::vector<float> row;
static int forwards = 0;

static void reset() {
    resident.clear();
    if (ctx) llama_memory_clear(llama_get_memory(ctx), true);
}
template<class F> static int attempt(F fn) {
    try { error.clear(); fn(); return 0; }
    catch (const std::exception & e) { error = e.what(); reset(); return -1; }
}
static void decode(const llama_token * ids, int n, int pos, int seq, bool logits) {
    auto batch = llama_batch_init(64, 0, 1);
    for (int start = 0; start < n; start += 64) {
        batch.n_tokens = std::min(64, n - start);
        for (int j = 0; j < batch.n_tokens; ++j) {
            batch.token[j] = ids[start + j];
            batch.pos[j] = pos + start + j;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq;
            batch.logits[j] = logits && start + j == n - 1;
        }
        ++forwards;
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("Bonsai decode failed (code " + std::to_string(rc) + ")");
        }
    }
    llama_batch_free(batch);
}

extern "C" {
EMSCRIPTEN_KEEPALIVE const char * bonsai_error() { return error.c_str(); }
EMSCRIPTEN_KEEPALIVE int bonsai_load(const char * path) {
    return attempt([&] {
        if (ctx) { llama_free(ctx); ctx = nullptr; }
        if (model) { llama_model_free(model); model = nullptr; }
        resident.clear();
        llama_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
        }, nullptr);
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
        mp.progress_callback = [](float, void *) { return true; };
        model = llama_model_load_from_file(path, mp);
        if (!model) throw std::runtime_error("Could not load Bonsai GGUF");
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.n_batch = cp.n_ubatch = 64;
        cp.n_seq_max = 2;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.kv_unified = true;
        cp.offload_kqv = cp.op_offload = false;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("Could not allocate Bonsai context");
    });
}
EMSCRIPTEN_KEEPALIVE int bonsai_tokenize(const char * text, int length, int special) {
    int count = 0;
    const int rc = attempt([&] {
        const auto * vocab = llama_model_get_vocab(model);
        const int n = -llama_tokenize(vocab, text, length, nullptr, 0, false, special);
        if (n < 0) throw std::runtime_error("Could not size Bonsai tokens");
        tokens.resize(n);
        count = llama_tokenize(vocab, text, length, tokens.data(), n, false, special);
        if (count < 0) throw std::runtime_error("Could not tokenize Bonsai input");
    });
    return rc < 0 ? rc : count;
}
EMSCRIPTEN_KEEPALIVE const llama_token * bonsai_tokens() { return tokens.data(); }
EMSCRIPTEN_KEEPALIVE int bonsai_prefix(const llama_token * ids, int n) {
    return attempt([&] {
        if (n < 1 || n >= 2048) throw std::runtime_error("State must fit the 2,048-token Bonsai context");
        if (resident.size() == size_t(n) && std::equal(resident.begin(), resident.end(), ids)) return;
        reset();
        decode(ids, n, 0, 0, false);
        resident.assign(ids, ids + n);
    });
}
EMSCRIPTEN_KEEPALIVE int bonsai_branch(const llama_token * ids, int n) {
    return attempt([&] {
        if (resident.empty()) throw std::runtime_error("No resident prefix");
        if (n < 1 || resident.size() + n > 2048) throw std::runtime_error("State + question exceeds the 2,048-token Bonsai context; shorten the input");
        const auto mem = llama_get_memory(ctx);
        llama_memory_seq_rm(mem, 1, -1, -1);
        llama_memory_seq_cp(mem, 0, 1, -1, -1);
        decode(ids, n, resident.size(), 1, true);
    });
}
// Copy just the requested label logits and candidate probability mass to JS.
EMSCRIPTEN_KEEPALIVE const float * bonsai_row(const int * labels, int n) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float max = *std::max_element(logits, logits + vocab);
    double all = 0, candidates = 0;
    for (int i = 0; i < vocab; ++i) all += std::exp(double(logits[i] - max));
    row.resize(n + 1);
    for (int i = 0; i < n; ++i) {
        row[i] = logits[labels[i]];
        candidates += std::exp(double(row[i] - max));
    }
    row[n] = candidates / all;
    return row.data();
}
EMSCRIPTEN_KEEPALIVE int bonsai_forwards() { const int n = forwards; forwards = 0; return n; }
}
