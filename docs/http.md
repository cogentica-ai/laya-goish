# Goish HTTP server

```sh
cd ~/projects/laya
./laya serve models/laya_english_f16.gguf --host 127.0.0.1 --port 8080 --threads 4
```

The server loads each configured GGUF once and runs inference in process.
Repeat `--model PATH` to load multiple backends; see [model registry](models.md). HTTP, goroutines,
signals and socket handling use Goish; the executable remains no_std Rust.

| Method | Path | Response |
|---|---|---|
| GET / HEAD | /health | Readiness, default model, loaded models, CPU device, busy state |
| GET / HEAD | /v1/models | Loaded models, aliases and default selection |
| GET / HEAD | /v1/presets | Ten built-in presets and their questions |
| POST | /v1/systemone | Typed decision |
| POST | /v1/decide | Alias for /v1/systemone |
| POST | /v1/decide/batch | Ordered results for shared questions across states |

```sh
curl -sS http://127.0.0.1:8080/health
curl -sS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/refund.json
```

Decision requests use the CLI/daemon schema:
`{"state": ..., "questions": {...}, "model": "laya", "id": "optional"}`.
English accepts `laya`, `english`, and `jev-latest`. OpenThai accepts
`openthai-systemone` when loaded. Requests route to their named model; omitted
or null `model` selects the configured default. Unknown names return 422.
Presets can also be
submitted, for example `{"preset":"guard","text":"Ignore previous instructions"}`.

Batch requests are `{"states":[...],"questions":{...},"model":"laya"}`, with a
`{"results":[...]}` response in the same order. The entire batch is validated
before inference. Limits: 256 states, 256 total questions, 1 MiB JSON body.
Laya supports 16 options per question. [OpenThai-SystemOne](openthai.md) supports
255 choice options or 2–10 score levels. Use `Content-Type: application/json`.

One inference request runs at a time across all loaded models, with `--threads` workers. Other inference
requests receive HTTP 429 and `Retry-After: 1`; health and metadata remain
available. Clients should retry with backoff. This avoids an unbounded work
queue and overlapping inference working sets. Batch states run sequentially.

Successful decisions include `X-Request-Id` and `X-Response-Time-Ms`.
Errors use `{"error":{"type":"...","message":"..."}}`.
Malformed JSON returns 400, invalid question/model schemas 422, unauthorized
requests 401, wrong methods 405, unsupported content types 415 and oversized
bodies 413. Raw model outputs are disabled unless the server starts with
`--raw`.

## Access and authentication

The default bind address is **127.0.0.1:8080**. To connect from a workstation:

Set `LAYA_API_KEY` (or fallback `TYPESAFE_API_KEY`) before starting the server to
require `Authorization: Bearer <key>` on decision and model-list routes.
Health, the root description and presets remain public. Keys are never logged.
`--host 0.0.0.0` explicitly listens on all IPv4 interfaces. TLS can terminate
at a reverse proxy; the built-in listener serves HTTP.

## Lifecycle and runtime limits

SIGINT/SIGTERM stops accepting connections and drains active requests for up
to 30 seconds. Requests still active at that deadline are terminated with the
process. Disconnected clients stop remaining work between question forwards;
a forward already running completes before cancellation is checked.

Goish alpha.13 buffers incoming bodies before calling the handler, with its
own 16 MiB parser limit. The application then enforces its 1 MiB API limit.
Connections are capped at 32, headers at 16 KiB, with header/read/idle deadlines
configured. No HTTP service manager or automatic reboot restart is installed.

Run the black-box integration suite with:

```sh
python3 validation/test_http.py
```

It starts temporary loopback servers, checks CLI parity, chunked input, error
handling, authentication, overload, disconnect cancellation and graceful
shutdown, then cleans up those servers.

## Memory soak test

Against the existing unauthenticated loopback server on port 8080, with its PID
in `server.pid` and the English F16 model loaded:

```sh
python3 validation/soak_http.py
python3 validation/soak_http.py --http-only
```

The default run checks inference against the CLI, then runs five rounds of
sequential inference, concurrent overload, mixed HTTP routes, and oversized
bodies. The HTTP-only run sends 100,000 mixed requests. Both alternate fresh
connections and keep-alive connections at 16-client concurrency, check expected
status codes, and observe 30 seconds of idle time. They sample RSS, anonymous
memory, swap, threads and file descriptors once per second, with detailed
`/proc/PID/smaps_rollup` at checkpoints. Reports are
`validation/http-soak.report.json` and `validation/http-churn.report.json`.
Load stops if host available memory falls below 700 MiB or server RSS grows
more than 3 GiB above its starting point.

These are bounded observations of resident memory and resource counts, not
proof that all allocations are freed or that every workload is leak-free.
