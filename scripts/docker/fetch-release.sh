#!/usr/bin/env bash
# Stages the `laya` executable from a GitHub release into dist/laya so the
# Dockerfile can COPY it. Requires `gh` authenticated against the private
# cogentica-ai/laya-goish repository.
set -euo pipefail
cd "$(dirname "$0")/../.."
version="${1:-v0.1.0}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

gh release download "$version" --repo cogentica-ai/laya-goish \
  --pattern "laya-goish-${version}-linux-x86_64.tar.gz" --pattern SHA256SUMS \
  --dir "$tmp"
(cd "$tmp" && sha256sum -c SHA256SUMS)
tar -xzf "$tmp/laya-goish-${version}-linux-x86_64.tar.gz" -C "$tmp"

mkdir -p dist
cp "$tmp/laya-goish-${version}-linux-x86_64/laya" dist/laya
chmod +x dist/laya
echo "Staged dist/laya from release ${version}"
