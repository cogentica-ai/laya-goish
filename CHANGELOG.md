# Release notes

## v0.1.0

The first Laya Goish release brings English and Thai decision models to a
single CPU inference server.

- Classify text, score it against ordered criteria, or estimate whether a
  condition is true. Use custom questions or ten built-in presets.
- Run Laya English from F16, Q8_0, or UD_Q4_K_M GGUF files.
- Convert and run OpenThai-SystemOne v0.3 for Thai and English decisions.
- Load both models and select one per request through a shared HTTP API.
- Use ordered batches, optional bearer authentication, health endpoints,
  overload responses, and graceful shutdown.
- Run the same models through the CLI or a persistent JSON-lines process.

### Download

`laya-goish-v0.1.0-linux-x86_64.tar.gz` includes a statically linked executable,
setup guides, request examples, model download/conversion scripts, and
attribution notices. Verify it with the accompanying `SHA256SUMS`.

Requires Linux x86-64 with AVX2/FMA. Model weights are downloaded separately.
See [getting started](docs/getting-started.md).

### Release limits

CPU inference only. One active inference request is allowed across all loaded
models; overlapping requests receive HTTP 429. Quantized Laya weights expand
to F32 for computation. OpenThai's validated format is F16, with a default
4,096-token total context and 2,048-token state budget. See the
[model guide](docs/openthai.md) for option limits and permutation costs.

### Release packaging

The v0.1.0 archive has been reissued with a pinned public Goish dependency,
remapped compiler source paths, stripped symbols, and normalized archive
ownership metadata. The inference implementation is unchanged.

### Validation

The inference revision passed CLI, HTTP, authentication, overload, shutdown,
and multi-model routing checks. Reference comparisons cover all three Laya
English formats and OpenThai tokenization and decisions. Numerical differences
from the reference engines are documented in the validation reports; bitwise
equivalence is not claimed.
