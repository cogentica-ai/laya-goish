#!/usr/bin/env python3
"""Build a stripped release without disclosing local source paths."""
import json
import os
import pathlib
import subprocess
import tomllib

from release_privacy import check_paths

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = "x86_64-unknown-linux-gnu"


def main():
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=ROOT, text=True,
    ))
    config = tomllib.loads((ROOT / ".cargo/config.toml").read_text())
    flags = list(config["target"][TARGET]["rustflags"])
    # Rust applies the last matching prefix, so put broad mappings first.
    mappings = {
        str(pathlib.Path.home()): "/build",
        str(pathlib.Path(os.environ.get("CARGO_HOME", pathlib.Path.home() / ".cargo")).resolve()): "/cargo",
        str(pathlib.Path(os.environ.get("RUSTUP_HOME", pathlib.Path.home() / ".rustup")).resolve()): "/rustup",
    }
    for package in metadata["packages"]:
        directory = pathlib.Path(package["manifest_path"]).parent.resolve()
        mappings[str(directory)] = "/deps/" + package["name"] + "-" + package["version"]
    mappings[str(ROOT)] = "/src/laya"
    for source, destination in sorted(mappings.items(), key=lambda item: len(item[0])):
        flags.append(f"--remap-path-prefix={source}={destination}")
    env = dict(os.environ)
    env.pop("RUSTFLAGS", None)
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    subprocess.run(["cargo", "build", "--release", "--locked"], cwd=ROOT, env=env, check=True)
    binary = ROOT / "target" / TARGET / "release" / "laya"
    check_paths(binary.read_bytes(), "release executable")
    print("Release build: stripped executable; private path scan passed")


if __name__ == "__main__":
    main()
