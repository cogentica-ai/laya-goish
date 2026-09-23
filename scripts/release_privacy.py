"""Reject common personal build paths in distributable files."""
import re

PRIVATE_PATHS = re.compile(
    rb"(?:/(?:home|Users)/[^/\s\x00]+|/mnt/[a-z]/(?:Users)/[^/\s\x00]+|"
    rb"[A-Za-z]:[\\/]Users[\\/][^\\/\s\x00]+)"
)


def check_paths(data, label):
    if PRIVATE_PATHS.search(data):
        # Do not echo the private value into build logs.
        raise RuntimeError(f"Personal filesystem path found in {label}; rebuild with scripts/build-release.py")
