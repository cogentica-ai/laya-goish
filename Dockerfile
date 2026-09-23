# syntax=docker/dockerfile:1
#
# Packages the statically linked Linux x86-64 release executable.
# AVX2/FMA must be available to the container, including under emulation.
# Stage dist/laya before building:
#   bash scripts/docker/fetch-release.sh v0.1.0

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
