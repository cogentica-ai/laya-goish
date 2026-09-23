# Inference design

This document describes the Laya backend; see [OpenThai-SystemOne](openthai.md)
for the separate Qwen3.5 backend.

The Laya runner specializes the ModernBERT graph exported by
`examples/laya/laya_trunk.py`. It does not execute arbitrary ggmlc programs.

The GGUF graph is needed even though execution is specialized:
- It identifies linear biases introduced by normalization folding.
- It identifies the remaining normalization weights and biases.
- It identifies full-attention versus sliding-attention layers.
- It supplies the weights, tokenizer, sequence limits and temperatures.

In particular, most layer-norm gamma/beta parameters have already been
folded into the following linear. Applying the original PyTorch affine
normalization again would be incorrect. The scorer has both a folded
normalization bias and its original bias through a fused bias/GELU node.

Only live tokens are evaluated. Omitting masked padding is equivalent to
the upstream padding mask for these noncausal attention layers. Head marker
indices retain the exact upstream sequence positions.

Matrix multiplication expands supported GGML quantized weights to F32,
uses four token rows per AVX2/FMA kernel, and divides disjoint output token
rows among Goish goroutines. The WaitGroup joins every worker before input
or output storage can be released.

## Goish alpha.14 preemption and AVX

Goish `1.0.0-alpha.14` preserves OS-enabled AVX and AVX-512 state with
`xsave64` / `xrstor64` during asynchronous preemption. Each suspended
goroutine keeps its register snapshot on its own stack, including when it
resumes on another worker.

Laya's SIMD kernels remain preemptible. The `acquirem()` / `releasem()`
workaround used with alpha.13 has been removed: that runtime saved XMM
registers but lost the upper YMM lanes, causing nondeterministic results
when AVX kernels were interrupted.

The target has no global AVX/native-CPU compilation flag: AVX2/FMA is
enabled only on the kernel function. The application verifies CPU and OS
AVX support before loading inference weights.

The daemon repeatability regression exercises identical requests separated
by invalid requests, and compares one-thread and four-thread outputs
exactly. This check is separate from approximate parity with GGML, which
uses lower-precision activation arithmetic for some tensor types.
