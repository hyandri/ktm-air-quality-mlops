"""Small helpers that say exactly which code and data produced a result."""
import hashlib
import subprocess

from hawa import config


def git_commit():
    """Current git commit hash, or "unknown" when git is not available."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             cwd=config.PROJECT_ROOT)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path):
    """SHA-256 of a file: changes if even one byte of the data changes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
