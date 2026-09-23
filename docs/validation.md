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

The Goish AVX/preemption guard and its regression are described in implementation.md. No Goish runtime source was changed.
