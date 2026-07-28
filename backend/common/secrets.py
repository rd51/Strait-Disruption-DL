"""
Credential loading, shared by every collector.

Written because the aisstream key cost real time to locate: it was saved as
`aisstream.txt` while the loader only looked for `aisstream.key`, and the
`.gitignore` matched `*.key` so a `.txt` credential would have been committed.
Both failure modes are handled here once, rather than re-invented per arm.

Resolution order, always: environment variable → files in `secrets/` → legacy
in-tree locations. Everything under `secrets/` is gitignored as a directory,
which protects by location instead of by filename pattern — a pattern only
protects the spellings someone remembered to list.
"""

from __future__ import annotations

import os
import string
import sys
from pathlib import Path


class SecretNotFound(RuntimeError):
    """No usable credential — the message names every location that was tried."""


# Re-exported so existing callers keep working. The implementation lives in
# paths.py and resolves the root by MARKER SEARCH, not by counting parents —
# counting is what broke when everything moved into backend/.
from .paths import repo_root, secrets_dir, ports_dir  # noqa: E402,F401


def safe_stdout() -> None:
    """
    Force UTF-8 on stdout/stderr, replacing anything unencodable.

    On Windows, Python picks the console codepage (cp1252 under Git Bash and
    when output is redirected). A single non-ASCII character in a *progress
    message* then raises UnicodeEncodeError and kills the job. That is exactly
    how a completed 14,592-slot backfill was lost to a `->` arrow: the download
    had finished, and the crash happened while printing the next line. Never
    let cosmetics take down a long job.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def looks_like_placeholder(value: str) -> bool:
    """
    Detect placeholder text of ANY shape, not one hardcoded string.

    The original AIS guard compared against the literal "PASTE_YOUR_KEY_HERE",
    so the placeholder actually on disk sailed past it and was sent to the
    server as though it were a real credential.
    """
    if not value:
        return True
    if set(value) <= set(string.ascii_uppercase + string.digits + "_"):
        return True
    lowered = value.lower()
    return any(w in lowered for w in
               ("paste", "your", "here", "xxxx", "changeme", "todo", "<", ">"))


def _read(path: Path) -> str:
    # utf-8-sig: Windows editors add a BOM to "plain text" routinely, and a BOM
    # silently corrupts a credential when it is sent to a server.
    return path.read_text(encoding="utf-8-sig").strip()


def load_secret(
    label: str,
    env_var: str,
    filenames: list[str],
    signup_url: str = "",
    min_len: int = 16,
) -> str:
    """
    Resolve a credential, or raise with an actionable message.

    `filenames` are looked for in `secrets/` first, then in `ports/` for
    backwards compatibility with credentials already sitting there.
    """
    env_value = os.environ.get(env_var, "").strip()
    if env_value and not looks_like_placeholder(env_value) and len(env_value) >= min_len:
        return env_value

    root = repo_root()
    searched: list[str] = []
    for directory in (root / "secrets", root / "ports"):
        for name in filenames:
            path = directory / name
            rel = path.relative_to(root)
            if not path.exists():
                searched.append(f"{rel}: not found")
                continue
            value = _read(path)
            if not value:
                searched.append(f"{rel}: empty")
            elif looks_like_placeholder(value):
                searched.append(f"{rel}: PLACEHOLDER TEXT ({len(value)} chars)")
            elif len(value) < min_len:
                searched.append(f"{rel}: too short ({len(value)} chars)")
            else:
                return value

    detail = "\n".join(f"      {s}" for s in searched)
    env_state = (
        "placeholder text" if env_value and looks_like_placeholder(env_value)
        else f"too short ({len(env_value)} chars)" if env_value
        else "unset"
    )
    raise SecretNotFound(
        f"No usable {label}.\n"
        f"    {env_var} env var: {env_state}\n"
        f"    files searched:\n{detail}\n"
        f"  Fix — put it in ONE of these (both are gitignored):\n"
        f"    1. secrets/{filenames[0]}\n"
        f"    2. the {env_var} line in .env\n"
        + (f"  Get one at: {signup_url}\n" if signup_url else "")
        + "  Never paste a credential into a chat, and never commit one."
    )


def redact(secret: str) -> str:
    """Safe-to-log form. Never log a credential itself."""
    if len(secret) <= 12:
        return "****"
    return f"{secret[:4]}…{secret[-4:]} ({len(secret)} chars)"
