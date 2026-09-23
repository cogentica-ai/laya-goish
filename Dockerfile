# syntax=docker/dockerfile:1
#
# Laya Goish ships only as a statically linked, freestanding Linux x86-64
# executable (no_std Rust, AVX2/FMA required). It cannot be built from
# source in this image: its `goish` dependency is an unpublished crate
# pinned to a path on the original author's machine (see Cargo.toml and
# docs/getting-started.md). This image packages the pre-built release
# binary instead, and runs it under linux/amd64 emulation (Rosetta 2 or
# QEMU) on Apple Silicon (M2) via Docker Desktop.
#
# Before building, stage the release binary at dist/laya, either with:
#   scripts/docker/fetch-release.sh v0.1.0
# (requires `gh` authenticated against the private cogentica-ai/laya-goish
# repo), or by manually extracting laya-goish-vX.Y.Z-linux-x86_64.tar.gz
# and copying its `laya` executable to dist/laya.

FROM --platform=linux/amd64 debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY dist/laya /app/laya
RUN chmod +x /app/laya

EXPOSE 8080
ENTRYPOINT ["/app/laya"]
CMD ["serve", "/models/laya_english_f16.gguf", "--host", "0.0.0.0", "--port", "8080", "--threads", "4"]
