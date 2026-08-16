// sparse-probe: P0 offline validator data capture for Quest-class sparse decode.
//
// Feeds a long corpus (-f corpus.txt) in chunks; at depth checkpoints
// (SPARSE_PROBE_CHECKPOINTS, comma-separated n_past values) runs
// SPARSE_PROBE_STEPS greedy single-token decode steps and captures, per
// attention layer, the q tensor consumed by FLASH_ATTN_EXT. After the final
// checkpoint it dumps the full K cache per layer, dequantized to f16 (values
// in the STORED basis as returned by the type's to_float).
//
// Output (SPARSE_PROBE_OUT dir): manifest.json + q_*.bin + K_layer*.f16.bin
//
// Run with prod-parity flags, e.g.:
//   SPARSE_PROBE_OUT=... SPARSE_PROBE_CHECKPOINTS=16384,32768,65536,126976 \
//   llama-sparse-probe -m model.gguf -c 131072 -ctk turbo4 -ctv turbo4 \
//     -fa on -ngl 99 --rope-scaling yarn --rope-scale 1.25 \
//     --yarn-orig-ctx 262144 -f corpus.txt --no-warmup

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct probe_state {
    // modes: 0 = off, 1 = capture q (probe step), 2 = dump K
    int mode = 0;
    int ckpt_idx = 0;
    int step_idx = 0;
    int fa_counter = 0; // resets per graph; counts FLASH_ATTN_EXT nodes -> layer order
    std::string outdir;
    FILE * manifest = nullptr;
    bool k_dumped = false;
};

static void dump_tensor_raw(struct ggml_tensor * t, const std::string & path) {
    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> buf(nbytes);
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
    FILE * f = fopen(path.c_str(), "wb");
    GGML_ASSERT(f);
    fwrite(buf.data(), 1, nbytes, f);
    fclose(f);
}

static void manifest_tensor(FILE * m, const char * kind, int ckpt, int step, int layer,
                            struct ggml_tensor * t, const char * file) {
    fprintf(m,
        "{\"kind\":\"%s\",\"ckpt\":%d,\"step\":%d,\"layer\":%d,\"name\":\"%s\",\"type\":\"%s\","
        "\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "],"
        "\"nb\":[%zu,%zu,%zu,%zu],\"file\":\"%s\"",
        kind, ckpt, step, layer, t->name, ggml_type_name(t->type),
        t->ne[0], t->ne[1], t->ne[2], t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3], file);
    // op params of the consumer are appended separately by caller when needed
    fprintf(m, "}\n");
    fflush(m);
}

static bool probe_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    probe_state * st = (probe_state *) user_data;
    if (ask) {
        return st->mode != 0 && t->op == GGML_OP_FLASH_ATTN_EXT;
    }
    if (st->mode == 0 || t->op != GGML_OP_FLASH_ATTN_EXT) {
        return true;
    }
    const int il = st->fa_counter++;

    struct ggml_tensor * q = t->src[0];
    struct ggml_tensor * k = t->src[1];

    if (st->mode == 1) {
        char fn[256];
        snprintf(fn, sizeof(fn), "q_c%d_s%d_l%02d.bin", st->ckpt_idx, st->step_idx, il);
        dump_tensor_raw(q, st->outdir + "/" + fn);
        // record FA op params (scale, max_bias, logit_softcap) once per node
        float params[4] = {0};
        memcpy(params, t->op_params, sizeof(params));
        fprintf(st->manifest,
            "{\"kind\":\"fa_params\",\"ckpt\":%d,\"step\":%d,\"layer\":%d,"
            "\"scale\":%g,\"max_bias\":%g,\"softcap\":%g,\"n_kv_view\":%" PRId64 "}\n",
            st->ckpt_idx, st->step_idx, il, params[0], params[1], params[2], k->ne[1]);
        manifest_tensor(st->manifest, "q", st->ckpt_idx, st->step_idx, il, q, fn);
    } else if (st->mode == 2 && !st->k_dumped) {
        // dump the K cache RAW quantized bytes (stored/WHT basis). Host-side
        // CPU traits->to_float must NOT be used: the CPU codec inverse-rotates
        // with a different table than CUDA (fork quirk) — python dequantizes
        // from raw with the CUDA formula (centroid table x block norm).
        struct ggml_tensor * ksrc = k->view_src ? k->view_src : k;
        char fn[256];
        snprintf(fn, sizeof(fn), "K_l%02d.raw.bin", il);
        const int64_t n_kv = k->ne[1];
        const size_t need = (size_t) n_kv * ksrc->nb[1];
        std::vector<uint8_t> buf(need);
        ggml_backend_tensor_get(ksrc, buf.data(), 0, need);
        FILE * f = fopen((st->outdir + "/" + fn).c_str(), "wb");
        GGML_ASSERT(f);
        fwrite(buf.data(), 1, need, f);
        fclose(f);
        fprintf(st->manifest,
            "{\"kind\":\"K\",\"layer\":%d,\"file\":\"%s\",\"n_rows\":%" PRId64 ",\"n_dims\":%" PRId64 ","
            "\"row_bytes\":%zu,\"src_type\":\"%s\","
            "\"view_ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "],"
            "\"view_nb\":[%zu,%zu,%zu,%zu]}\n",
            il, fn, n_kv, ksrc->ne[0], ksrc->nb[1], ggml_type_name(ksrc->type),
            k->ne[0], k->ne[1], k->ne[2], k->ne[3],
            k->nb[0], k->nb[1], k->nb[2], k->nb[3]);
        fflush(st->manifest);
    }
    return true;
}

static std::vector<int> parse_ckpts(const char * s) {
    std::vector<int> out;
    if (!s) return out;
    std::string cur;
    for (const char * p = s;; ++p) {
        if (*p == ',' || *p == 0) {
            if (!cur.empty()) out.push_back(atoi(cur.c_str()));
            cur.clear();
            if (*p == 0) break;
        } else {
            cur += *p;
        }
    }
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    probe_state st;
    const char * out = getenv("SPARSE_PROBE_OUT");
    st.outdir = out ? out : "sparse-probe-out";
    std::string mkcmd = "mkdir -p " + st.outdir;
    if (system(mkcmd.c_str()) != 0) { LOG_ERR("mkdir failed\n"); return 1; }

    std::vector<int> ckpts = parse_ckpts(getenv("SPARSE_PROBE_CHECKPOINTS"));
    if (ckpts.empty()) ckpts = {16384, 32768, 65536};
    const char * steps_env = getenv("SPARSE_PROBE_STEPS");
    const int n_steps = steps_env ? atoi(steps_env) : 4;

    params.cb_eval = probe_cb;
    params.cb_eval_user_data = &st;
    params.warmup = false;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    LOG_INF("corpus tokens: %zu\n", tokens.size());

    st.manifest = fopen((st.outdir + "/manifest.jsonl").c_str(), "w");
    GGML_ASSERT(st.manifest);
    fprintf(st.manifest, "{\"kind\":\"run\",\"n_ctx\":%d,\"checkpoints\":[", (int) llama_n_ctx(ctx));
    for (size_t i = 0; i < ckpts.size(); ++i) fprintf(st.manifest, "%s%d", i ? "," : "", ckpts[i]);
    fprintf(st.manifest, "],\"n_steps\":%d,\"corpus_tokens\":%zu}\n", n_steps, tokens.size());

    llama_sampler * smpl = llama_sampler_init_greedy();

    int n_past = 0;
    size_t consumed = 0;
    const int chunk = 1024;
    const int n_ctx = (int) llama_n_ctx(ctx);

    for (size_t ci = 0; ci < ckpts.size(); ++ci) {
        const int target = ckpts[ci];
        if (target >= n_ctx - 8 - n_steps * (int) ckpts.size()) {
            LOG_ERR("checkpoint %d too close to n_ctx %d\n", target, n_ctx);
            break;
        }
        // prefill up to target
        while (n_past < target && consumed < tokens.size()) {
            int n = std::min((int) (tokens.size() - consumed), std::min(chunk, target - n_past));
            llama_batch batch = llama_batch_get_one(tokens.data() + consumed, n);
            if (llama_decode(ctx, batch)) { LOG_ERR("decode failed at %d\n", n_past); return 1; }
            consumed += n;
            n_past += n;
        }
        if (n_past < target) { LOG_ERR("corpus exhausted at %d < %d\n", n_past, target); break; }
        LOG_INF("checkpoint %d reached (n_past=%d), probing %d steps\n", target, n_past, n_steps);

        // probe: greedy decode n_steps tokens with q capture
        st.ckpt_idx = (int) ci;
        for (int s = 0; s < n_steps; ++s) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (tok == llama_vocab_eos(vocab)) tok = llama_vocab_nl(vocab) >= 0 ? llama_vocab_nl(vocab) : tok;
            st.mode = 1; st.step_idx = s; st.fa_counter = 0;
            llama_batch batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(ctx, batch)) { LOG_ERR("probe decode failed\n"); return 1; }
            st.mode = 0;
            n_past += 1;
        }
        fprintf(st.manifest, "{\"kind\":\"ckpt_done\",\"ckpt\":%d,\"n_past\":%d}\n", (int) ci, n_past);
        fflush(st.manifest);
    }

    // final: dump K at the deepest point
    {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        st.mode = 2; st.fa_counter = 0;
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, batch)) { LOG_ERR("kdump decode failed\n"); return 1; }
        st.mode = 0;
        st.k_dumped = true;
        n_past += 1;
        fprintf(st.manifest, "{\"kind\":\"kdump_done\",\"n_past\":%d}\n", n_past);
    }

    fclose(st.manifest);
    LOG_INF("probe complete, n_past=%d, out=%s\n", n_past, st.outdir.c_str());
    llama_backend_free();
    return 0;
}
