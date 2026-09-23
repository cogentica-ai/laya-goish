#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
for quant in "${@:-f16}"; do
 case "$quant" in f16|q8_0|ud_q4_k_m) ;; *) echo "Unsupported quant: $quant" >&2; exit 1;; esac
 file="laya_english_${quant}.gguf"
 if [ ! -f "models/$file" ]; then
  curl --fail --location --retry 3 -o "models/$file.part" "https://huggingface.co/mys/laya-GGUF/resolve/main/$file"
  mv "models/$file.part" "models/$file"
 fi
done
