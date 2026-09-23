# Validation results

Validated on Linux x86-64 with four CPU threads.

All comparisons use the original ggmlc source revision recorded in provenance.json.

| GGUF | Questions | Maximum raw logit difference | Maximum probability difference |
|---|---:|---:|---:|
| models/laya_english_f16.gguf | 6 | 0.008081 | 0.000927 |
| models/laya_english_q8_0.gguf | 6 | 0.133957 | 0.009319 |
| models/laya_english_ud_q4_k_m.gguf | 6 | 0.115709 | 0.013061 |

All 18 decision comparisons passed. Token IDs and option-marker positions match exactly. The cases cover choice, score, noul, Unicode, MASK-token sanitization, and 512-token truncation.

PASS: self-tests, 10 presets, invalid CLI/GGUF, daemon recovery, repeated requests, 1/4-thread parity, numeric-state parity

PASS: 13 tokenizer parity cases

The binary is statically linked and has no dynamic section. The historical differential-run binary checksum is recorded in validation/binary-sha256.txt; model checksums are in validation/model-sha256.txt. Current release binary checksums are recorded in the archive's BUILD.json.

A 45-token refund request took about 1.2 seconds with four threads in the differential run. Timing varies with other workloads; these are correctness runs, not isolated performance benchmarks.

The largest quantized-model probability difference was about 0.0131 (1.31 percentage points). Rust expands weights to F32, whereas GGML quantizes some activations during matrix multiplication. Exact one-thread/four-thread and repeated-request equality was separately verified for the Rust implementation.

These original validation results used the alpha.13 AVX/preemption guard.
Current source uses Goish alpha.14's XSAVE/XRSTOR support instead; see
[implementation details](implementation.md#goish-alpha14-preemption-and-avx).

## Goish alpha.14 upgrade

The optimized build passed self-tests and the release builder's private-path
scan with Goish `1.0.0-alpha.14` from crates.io. The CLI smoke suite and
multi-model HTTP registry suite also passed, including routing, aliases,
authentication, overload handling and graceful shutdown.

The preemption regression checked Laya FP16, Q8_0, Q4_K_M and OpenThai
SystemOne FP16 with one and four threads. All 64 repeated inferences under
forced SIGURG signals matched the previous alpha.13 binary exactly, including
raw logits; only request latency was excluded from the comparison. SIMD
kernels ran without the previous preemption guard.

See the [recorded results](../validation/goish-alpha14-preemption.report.json).
To repeat this check with models installed:

```sh
python3 validation/preemption.py --baseline /path/to/previous/laya
```

Omit `--baseline` to compare against the current binary's one-thread output.
Signal delivery counts record stress intensity, not completed context switches.
