#!/usr/bin/env python3
"""Package the built Linux executable and supporting files for a GitHub release."""
import hashlib
import json
import pathlib
import shutil
import subprocess
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = "x86_64-unknown-linux-gnu"


def output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def main():
    metadata = json.loads(output("cargo", "metadata", "--locked", "--format-version", "1"))
    package = next(p for p in metadata["packages"] if p["name"] == "laya")
    version = package["version"]
    name = f"laya-goish-v{version}-linux-x86_64"
    destination = ROOT / "target" / "release-dist"
    stage = destination / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    binary = ROOT / "target" / TARGET / "release" / "laya"
    shutil.copy2(binary, stage / "laya")
    subprocess.run(["strip", "--strip-unneeded", str(stage / "laya")], check=True)

    paths = output(
        "git", "ls-files", "--cached", "--others", "--exclude-standard", "--",
        "README.md", "CHANGELOG.md", "NOTICE", "LICENSES", "docs", "examples",
        "scripts", "presets.json", "validation"
    ).splitlines()
    for relative in paths:
        source = ROOT / relative
        if source.is_file():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    dependencies = []
    for dependency in metadata["packages"]:
        if dependency["id"] == package["id"]:
            continue
        directory = pathlib.Path(dependency["manifest_path"]).parent
        notices = [
            path for path in directory.iterdir()
            if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING", "NOTICE"))
        ]
        # The macro crate shares the license at the Goish workspace root.
        if not notices and dependency["name"] == "goish-macros":
            notices = [directory.parent / "LICENSE"]
        if not notices:
            raise RuntimeError(f"No license file found for {dependency['name']}")
        target = stage / "LICENSES" / "dependencies" / (
            dependency["name"] + "-" + dependency["version"]
        )
        target.mkdir(parents=True)
        for notice in notices:
            shutil.copy2(notice, target / notice.name)
        dependencies.append({
            "name": dependency["name"], "version": dependency["version"],
            "license": dependency.get("license"),
        })

    ggml_notice = ROOT / "upstream" / "third_party" / "ggml" / "LICENSE"
    if not ggml_notice.is_file():
        raise RuntimeError("Missing upstream/third_party/ggml/LICENSE; restore the pinned upstream checkout.")
    shutil.copy2(ggml_notice, stage / "LICENSES" / "ggml-MIT.txt")
    manifest = {
        "name": package["name"], "version": version, "target": TARGET,
        "source_commit": output("git", "rev-parse", "HEAD"),
        "source_dirty": bool(output("git", "status", "--porcelain")),
        "rustc": output("rustc", "--version"),
        "binary_sha256": sha256(stage / "laya"),
        "dependencies": dependencies,
    }
    (stage / "BUILD.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = destination / (name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(stage, arcname=name)
    (destination / "SHA256SUMS").write_text(sha256(archive) + "  " + archive.name + "\n")
    print(archive)
    print(destination / "SHA256SUMS")


if __name__ == "__main__":
    main()
