# Get started

## Install the release

The v0.1.0 executable requires Linux x86-64 with AVX2 and FMA. It is statically
linked and runs CPU inference without Python or an external inference runtime.
Use a machine with at least 4 GB of available memory for Laya English F16;
allow more for longer inputs or multiple loaded models.

The repository is private. With access to it and an authenticated GitHub CLI:

```sh
gh release download v0.1.0 --repo cogentica-ai/laya-goish \
  --pattern 'laya-goish-v0.1.0-linux-x86_64.tar.gz' --pattern SHA256SUMS
sha256sum -c SHA256SUMS
tar -xzf laya-goish-v0.1.0-linux-x86_64.tar.gz
cd laya-goish-v0.1.0-linux-x86_64
./laya self-test
bash scripts/download-models.sh f16
./laya serve models/laya_english_f16.gguf --port 8080 --threads 4
```

The model download requires curl and approximately 807 MiB of disk space.
Weights have their publisher's terms and are not included in the release.

In another terminal, from the same directory:

```sh
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/v1/decide \
  -H 'Content-Type: application/json' --data-binary @examples/refund.json
```

The server listens on loopback by default. See [HTTP access and authentication](http.md#access-and-authentication)
for remote access, bearer tokens, batches, and lifecycle behavior.

## Run with Docker (including Apple Silicon / M2)

The executable requires Linux x86-64 with AVX2/FMA and cannot currently be
built for arm64 (its `goish` dependency is an unpublished crate; see
[build from source](#build-from-source)). On an Apple Silicon Mac, Docker
Desktop runs `linux/amd64` images through Rosetta 2 or QEMU emulation, which
is enough to run the statically linked release binary.

Stage the release binary once, either with `gh` against the private repo:

```sh
scripts/docker/fetch-release.sh v0.1.0
```

or by manually extracting a release tarball and copying its `laya`
executable to `dist/laya`. Then:

```sh
docker compose --profile tools run --rm download-models f16
docker compose up --build
curl http://127.0.0.1:8080/health
```

`docker-compose.yml` binds `./models` into the container read-only and
`./scripts` for the model-download helper. Edit the `laya` service's
`command:` to load additional models or change ports/threads; set
`LAYA_API_KEY` in the environment to require bearer authentication.

## Use a preset

Ten presets provide ready-made questions. List them or run one from the CLI:

```sh
./laya list-presets
./laya decide models/laya_english_f16.gguf \
  --preset email --text "I was charged twice. Please refund me."
```

For a custom request, use `--request examples/refund.json`. To keep the model
loaded while processing one JSON request per line:

```sh
./laya daemon models/laya_english_f16.gguf --threads 4 < examples/refund.jsonl
```

Requests contain a `state` string or JSON value and an ordered `questions`
object. Question types are `choice` (named criteria), `score` (ordered criteria),
and `noul` (a yes/no proposition). The API accepts at most 256 questions per
request. Laya supports up to 16 options per question; OpenThai has different
limits documented in its [model guide](openthai.md).

## Add Thai and English decisions

Follow the [OpenThai conversion guide](openthai.md#convert-the-pinned-checkpoint)
to create `models/openthai_systemone_v03_f16.gguf`. Python is required for
conversion only. Then load both models:

```sh
./laya serve models/laya_english_f16.gguf \
  --model models/openthai_systemone_v03_f16.gguf \
  --default-model laya --port 8080 --threads 4
```

Send `"model": "openthai-systemone"` to select OpenThai. Omitting `model`
selects the default. See [model routing](models.md) and the
[Thai example](../examples/openthai-thai.json).

## Quantized models and memory

Laya English F16, Q8_0, and UD_Q4_K_M files are supported:

```sh
bash scripts/download-models.sh f16 q8_0 ud_q4_k_m
```

The loader decodes F32, F16, Q4_0, Q8_0, Q4_K, and Q6_K tensors.
Matrix weights expand to F32 at load time: quantization reduces download and
disk size, but inference does not use native quantized matrix kernels.
English weights use approximately 1.7 GB before mapped files and activations.
OpenThai F16 is the validated conversion; its embeddings stay mapped and
only used rows are decoded.

One inference request runs at a time across the server. Other concurrent
inference requests receive HTTP 429 with `Retry-After: 1`. Batch items run
sequentially. CPU time and memory needs grow with input length, question count,
and OpenThai option permutations. There is no GPU backend in this release.

## Build from source

Install Rust/Cargo and check out Goish at the revision recorded in
[provenance.json](provenance.json). The Goish dependency in `Cargo.toml`
currently uses the original build machine's absolute path; update it to
your local Goish checkout before building.

```sh
cargo build --release --locked
./target/x86_64-unknown-linux-gnu/release/laya self-test
```

The repository's `./laya` symlink points to that release executable.
See [implementation details](implementation.md), [validation](validation.md),
and [OpenThai reference checks](openthai.md#reference-validation).

To package a release after building, use Python 3, Cargo, and GNU strip.
The pinned ggmlc checkout must also be present at `upstream/` so its ggml
license notice can be included:

```sh
python3 scripts/package-release.py
```

The archive and checksums are written to `target/release-dist/`. Packaging
copies and strips the executable, collects dependency notices, and records
the source revision and binary checksum in `BUILD.json`.
