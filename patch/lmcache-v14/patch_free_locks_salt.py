#!/usr/bin/env python3
"""
Patch: pass cache_salt when freeing lookup locks (connector -> adapter).

Bug found 2026-07-02: LMCacheMPConnector.update_state_after_alloc() calls
scheduler_adapter.free_lookup_locks() WITHOUT cache_salt, so the adapter
builds an IPCCacheEngineKey with cache_salt="" and the server-side
free_lookup_locks() releases locks against UNSALTED ObjectKeys. For any
request that used a non-empty cache_salt, the real (salted) L1 entries
keep their read locks forever, and the server floods:

    L1Manager: finish read on non-existing key ObjectKey(...,
    cache_salt=''), potential inconsistent data might be read

Observed: 52 leaked read locks per 14k-token salted request (GLM-5.2 v12,
DCP=4, chunk=1024).

Fix: one-line — pass cache_salt=tracker.cache_salt at the call site. The
adapter signature already accepts it; every other _create_key path in the
adapter propagates the salt correctly.

Idempotent: checks for FREE-LOCKS-SALT-FIX marker. Exits non-zero on
match failure so `set -e` startup aborts loudly instead of silently
running unpatched.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

MARKER = "FREE-LOCKS-SALT-FIX"
TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/lmcache/"
    "integration/vllm/lmcache_mp_connector.py"
)

OLD = """                if free_end > 0:
                    self.scheduler_adapter.free_lookup_locks(
                        token_ids=list(tracker.all_token_ids),
                        start=0,
                        end=free_end,
                        request_id=request.request_id,
                    )"""

NEW = """                if free_end > 0:
                    self.scheduler_adapter.free_lookup_locks(
                        token_ids=list(tracker.all_token_ids),
                        start=0,
                        end=free_end,
                        request_id=request.request_id,
                        cache_salt=tracker.cache_salt,  # """ + MARKER + """
                    )"""


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    text = target.read_text()

    if MARKER in text:
        print(f"[{MARKER}] already patched, skipping")
        return 0

    if OLD not in text:
        print(f"[{MARKER}] ERROR: free_lookup_locks call site not found in {target}")
        return 1

    text = text.replace(OLD, NEW, 1)
    target.write_text(text)
    py_compile.compile(str(target), doraise=True)
    print(f"[{MARKER}] applied: free_lookup_locks now passes tracker.cache_salt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
