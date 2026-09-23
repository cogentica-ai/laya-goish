# OpenThai-SystemOne on Goish

This backend runs [iapp/OpenThai-SystemOne](https://huggingface.co/iapp/OpenThai-SystemOne)
v0.3 directly in the no_std Goish Rust executable, using GGUF v3.
It is a separate architecture from Laya/ModernBERT. It does not invoke Python,
Transformers, GGML, or another inference process at runtime.

## Convert the pinned checkpoint

Python is needed only for conversion and reference validation:

```sh
python3 -m venv .venv
.venv/bin/pip install -r scripts/requirements-openthai.txt
.venv/bin/python scripts/convert-openthai.py --download
```

The converter downloads the official checkpoint at
`f3709948b5e3cc9606a57e74ba62b7a639d17dd3`, verifies the source weights,
tokenizer and configuration against that revision, and writes
`models/openthai_systemone_v03_f16.gguf`. Matrix tensors use F16; vectors
use F32. Metadata contains the complete original configuration, vocabulary,
merges, special-token types, and source revision. Calibration weights are
preserved in the log-temperature tensor.
Tensor names follow the source checkpoint; `general.architecture` is
`openthai_systemone`. This is our engine's GGUF architecture, not a claim
of compatibility with llama.cpp or the ggmlc Laya graph executor.

The converter reports its largest weight conversion error and output SHA-256
in `validation/openthai-conversion.report.json`. Downloaded checkpoints,
models, Python environments and build artifacts remain outside Git.

## Build and run

The existing Goish dependency and Linux x86-64 AVX2/FMA requirements apply:

```sh
cargo build --release --locked
./laya info models/openthai_systemone_v03_f16.gguf
./laya decide models/openthai_systemone_v03_f16.gguf --request examples/openthai-thai.json
./laya serve models/openthai_systemone_v03_f16.gguf --port 8081 --threads 4
curl http://127.0.0.1:8081/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @examples/openthai-thai.json
```

CLI `decide`, `daemon`, `tokenize`, and the existing HTTP routes select
the backend from GGUF metadata. HTTP model name: `openthai-systemone`.
A server can load both Laya and OpenThai using `--model PATH` and route by the
request's `model` field; see [multi-model serving](models.md). Separate processes
and ports are also supported.
Authentication and overload behavior are the same as described in [http.md](http.md).

All questions in an OpenThai request share one causal forward pass, with
outputs read at each answer marker. Supported question types:

- `choice`: 1–255 named options, probabilities, entropy confidence, and abstain probability.
- `score`: 2–10 ordered levels, weighted score, legend, probabilities and confidence.
- `noul`: probability of yes; optional false/true descriptions.

The response follows the upstream OpenThai schema, which differs from Laya's
optional action fields. `usage` also reports latency, permutation count,
and whether the state was truncated.

For choices with more than ten options, the default averages eight cyclic
option orders, matching the upstream client. `"permutations": 1` or
`"order_invariant": false` disables this; explicit `permutations` accepts
1–32 and takes precedence. Permutations run sequentially on this CPU backend,
so latency grows with their count. HTTP batch requests preserve these settings.

## Implementation and limits

The implementation includes all 24 Qwen3.5 text layers: 18 Gated-DeltaNet
layers with causal convolution and recurrent state, six causal grouped-query
attention layers with partial RoPE and output gates, zero-centered RMSNorm,
SwiGLU, the 256-slot head and per-type temperatures. Recurrent state is fresh
for every forward. F32 matrix kernels use the existing Goish preemption guard.

The tokenizer uses the published Unicode pretokenization, NFC normalization,
byte-level BPE and explicit added tokens. Formatting preserves question/option
order and sanitizes injected TypeSafe control tokens.

This version specializes the published 0.8B v0.3 configuration and rejects
incompatible configurations. It is not a general Qwen language-model runner.

The default CPU context limit is **4,096 tokens**, including all questions;
state truncation keeps the state marker and the tail of the state sequence
within a **2,048-token** state budget. Oversized question blocks are rejected. The converter
can set `--context 32..8192`. This implementation does not expose the
upstream client's 64k-token limit: full attention here is quadratic and large
contexts are expensive. Up to 256 questions and a 1 MiB HTTP body are allowed.

Matrix weights expand to F32 at load time; embeddings stay mapped in F16 and
only used rows are decoded. There is no GPU path or cross-request cache.

## Reference validation

Install the pinned CPU reference dependencies separately:

```sh
.venv/bin/pip install -r validation/requirements-openthai.txt
.venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python validation/openthai_reference.py
.venv/bin/python validation/openthai_tokenizer.py
python3 validation/openthai_check.py
.venv/bin/python validation/openthai_edges.py
python3 validation/test_openthai_http.py
```

The reference script downloads and executes the published model wrapper and
client at pinned revisions, using CPU/F32 Transformers 5.17.0. It is test-only.
Checks compare exact token IDs and answer positions, raw logits, final
probabilities, scores, confidence, abstain, and permutation counts.
See `validation/openthai.report.json` and
`validation/openthai-tokenizer.report.json` for measured results.
