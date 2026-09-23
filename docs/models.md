# Model backends and multi-model serving

Source code and model weights have separate directories:

```text
src/
  models/
    mod.rs                 Backend trait and GGUF architecture dispatch
    laya/
      mod.rs               Backend implementation
      model.rs             ModernBERT inference
      decision.rs          Laya sequence formatting and decision decoding
    openthai/
      mod.rs               Backend implementation
      model.rs             Qwen3.5 / Gated-DeltaNet inference
      decision.rs          OpenThai sequence formatting and slot decoding
  engine.rs                One backend, its tokenizer and identity
  registry.rs              Loaded models, aliases and default selection
  request.rs               Shared preset expansion
  server.rs                HTTP transport, authentication and admission
  gguf.rs                  Shared GGUF reader and tensor decoding
  tokenizer.rs             Shared tokenizer implementations
  tensor.rs                Shared CPU kernels
models/                    Local GGUF weights, excluded from Git
```

The `Backend: Send + Sync` trait exposes validation, layer count and decision
inference. Each backend owns immutable weights; its request state is local to
each forward. `Engine` owns the backend and corresponding tokenizer.
The registry owns each engine once and maps aliases to that same instance.

## Start one server with both models

```sh
cd /home/chanwit/projects/laya
./laya serve models/laya_english_f16.gguf \
  --model models/openthai_systemone_v03_f16.gguf \
  --default-model laya \
  --host 127.0.0.1 --port 8080 --threads 4
```

Repeat `--model PATH` for each additional GGUF. The positional model is the
default unless `--default-model NAME` specifies another canonical name or alias.
All selected models load before the HTTP listener opens. Models remain loaded
for the process lifetime; there is no on-demand loading or hot reload.

Two checkpoints with the same canonical name or colliding aliases are rejected,
including loading English Laya F16 and Q8_0 together under the name `laya`.
Unknown architectures, missing files and an unknown default also prevent startup.
The registry preflights these configuration errors before weight expansion.

The server loads only explicitly configured files. It does not scan the weights
directory or interpret a request's model name as a file path.

## Request routing

```sh
curl http://127.0.0.1:8080/v1/models

curl http://127.0.0.1:8080/v1/decide \
  -H 'Content-Type: application/json' \
  --data '{"model":"laya","state":"Please refund my payment.","questions":{"refund":{"type":"noul","instructions":"Does the customer ask for money back?"}}}'

curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data '{"model":"openthai-systemone","state":"ลูกค้าขอเงินคืน","questions":{"refund":{"type":"noul","instructions":"ลูกค้าขอเงินคืนหรือไม่"}}}'
```

Omitting `model` (or setting it to JSON null) selects the configured default.
English Laya accepts `laya`, `english` and `jev-latest`; OpenThai accepts
`openthai-systemone`. Unknown names and non-string non-null values return 422.
Responses identify the canonical model.

`GET /v1/models` lists every available name, its canonical model, architecture,
family and default flag, plus top-level `default_model`.
`GET /health` keeps the existing `model` and `family` fields for the default
model and adds `models`, the canonical names of all loaded models.

Both `/v1/decide` and `/v1/systemone` route this way.
`/v1/decide/batch` selects one model for the whole batch using its top-level
`model` field. Validation, token budgets, option limits, formatting and output
decoding come from the selected backend. The same authentication policy applies
to every backend.

## Concurrency and memory

The existing admission limit is **one active inference request across the entire
server**, including batches and requests targeting different models. A concurrent
request returns 429 with `Retry-After: 1`. Health and metadata remain responsive.
This is multi-model residency and routing, not simultaneous inference execution.

`--threads` configures the process-wide Goish scheduler and the CPU worker count
used by each backend. Resident weights add across models; choose the set of
loaded checkpoints to fit the host. All backends share the HTTP server's
30-second graceful-shutdown deadline.

## Extending the engine

Add a directory under `src/models/`, implement `Backend`, and register the
GGUF identity and constructor in `src/models/mod.rs`. Keep model-specific
formatting and decoding inside that backend. The CLI, registry and HTTP transport
use the common interface.

Single-model `info`, `tokenize`, `decide`, `daemon` and `serve` commands
remain supported. Additional models and default selection are serving options.

Run the real-model routing/lifecycle regression with:

```sh
python3 validation/test_registry.py
```

It starts temporary servers, checks CLI parity for both models, aliases/defaults,
backend-specific limits, batches, authentication, overload and graceful shutdown,
then closes the servers. Results are written to `validation/registry.report.json`.
