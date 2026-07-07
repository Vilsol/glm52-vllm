#!/usr/bin/env python3
"""Patch fs_native L2 adapter to scan existing .data files on startup.

Why:
- Native FS L2 reads existing cache files fine, but NativeConnectorL2Adapter only
  accounts newly-stored keys.
- After a restart, /status reported total_bytes_used=0 even though L2 files existed
  and L2 hits worked.
- That makes L2 eviction blind to pre-existing disk usage. The Python fs adapter
  already has startup scan + listener replay; this patch adds the equivalent to
  fs_native.
"""
from __future__ import annotations

from pathlib import Path

NATIVE_PATH = Path('/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py')
FS_NATIVE_PATH = Path('/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/l2_adapters/fs_native_l2_adapter.py')
MARKER_NATIVE = '[FS-NATIVE-STARTUP-SCAN native v1]'
MARKER_FACTORY = '[FS-NATIVE-STARTUP-SCAN factory v1]'


def patch_native_connector() -> bool:
    text = NATIVE_PATH.read_text()
    if MARKER_NATIVE in text:
        print('patch_fs_native_startup_scan: native connector already patched')
        return False

    old = '''        # Pending store sizes: native future_id -> (keys, per_key_sizes).
        # Bridges the async store submit → demux completion gap so the
        # demux thread can fire ``_notify_keys_stored(keys, sizes)``.
        self._pending_store_sizes: dict[int, tuple[list[ObjectKey], list[int]]] = {}
'''
    new = old + f'''
        # Startup scan state for filesystem-native backends. Populated by
        # prime_existing_keys() after the factory scans existing .data files.
        # {MARKER_NATIVE}
        self._scanned_keys: list[ObjectKey] = []
        self._scanned_sizes: list[int] = []
        self._scan_done: bool = False
'''
    if old not in text:
        raise SystemExit('native connector insertion point not found (pending_store_sizes)')
    text = text.replace(old, new, 1)

    old = '''    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class tracks aggregate + per-user totals via ``_notify_keys_*``;
    # we feed it the byte sizes from each store/delete completion.

    # ---------------------------------------------------------------
    # Status Interface
    # ---------------------------------------------------------------
'''
    new = '''    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class tracks aggregate + per-user totals via ``_notify_keys_*``;
    # we feed it the byte sizes from each store/delete completion.

    def prime_existing_keys(
        self,
        keys: list[ObjectKey],
        sizes: list[int],
    ) -> None:
        """Prime byte/LRU accounting for files that already exist on disk.

        Called by fs_native factory after scanning base_path. This mirrors
        FSL2Adapter's startup scan behavior: byte counters are updated
        immediately, and scanned keys are replayed to listeners registered
        later by StorageManager.
        """
        if len(keys) != len(sizes):
            raise ValueError("keys and sizes length mismatch")

        # De-duplicate while preserving order. If a key already exists in
        # _key_sizes (should not happen during startup), keep first size.
        unique_keys: list[ObjectKey] = []
        unique_sizes: list[int] = []
        seen: set[ObjectKey] = set()
        for key, size in zip(keys, sizes, strict=True):
            if key in seen:
                continue
            seen.add(key)
            unique_keys.append(key)
            unique_sizes.append(int(size))

        with self._lock:
            self._key_sizes = dict(zip(unique_keys, unique_sizes, strict=True))
            self._scanned_keys = unique_keys
            self._scanned_sizes = unique_sizes
            self._scan_done = True

        by_salt: dict[str, int] = {}
        total = 0
        for key, size in zip(unique_keys, unique_sizes, strict=True):
            by_salt[key.cache_salt] = by_salt.get(key.cache_salt, 0) + size
            total += size
        with self._usage_lock:
            self._total_bytes_used = total
            self._bytes_by_cache_salt = by_salt

        if unique_keys:
            logger.info(
                "FSNative startup scan primed %.2f GB in %d existing files",
                total / 1e9,
                len(unique_keys),
            )

    def register_listener(self, listener) -> None:
        """Register listener and replay startup-scanned keys. [FS-NATIVE-STARTUP-SCAN native v1]"""
        super().register_listener(listener)
        with self._lock:
            scanned = list(self._scanned_keys) if self._scan_done else []
        if scanned:
            listener.on_l2_keys_stored(scanned)

    # ---------------------------------------------------------------
    # Status Interface
    # ---------------------------------------------------------------
'''
    if old not in text:
        raise SystemExit('native connector insertion point not found (Status Interface)')
    text = text.replace(old, new, 1)

    backup = NATIVE_PATH.with_suffix('.py.bak_fs_native_scan')
    if not backup.exists():
        backup.write_text(NATIVE_PATH.read_text())
    NATIVE_PATH.write_text(text)
    print('patch_fs_native_startup_scan: patched native connector')
    return True


def patch_fs_native_factory() -> bool:
    text = FS_NATIVE_PATH.read_text()
    if MARKER_FACTORY in text:
        print('patch_fs_native_startup_scan: fs_native factory already patched')
        return False

    old = 'from typing import TYPE_CHECKING, Optional\n'
    new = old + 'from pathlib import Path\n'
    if old not in text:
        raise SystemExit('fs_native import insertion point not found')
    text = text.replace(old, new, 1)

    old = 'logger = init_logger(__name__)\n\n\nclass FSNativeL2AdapterConfig'
    new = '''logger = init_logger(__name__)


def _scan_existing_fs_native_files(base_path: str):
    """Return (keys, sizes) for existing .data files. [FS-NATIVE-STARTUP-SCAN factory v1]"""
    from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
        _filename_to_object_key,
    )

    keys = []
    sizes = []
    root = Path(base_path)
    try:
        iterator = root.glob("*.data")
    except Exception:
        logger.exception("FSNative startup scan: failed to open %s", base_path)
        return keys, sizes

    for path in iterator:
        try:
            if not path.is_file():
                continue
            key = _filename_to_object_key(path.name)
            if key is None:
                continue
            size = path.stat().st_size
            if size <= 0:
                continue
            keys.append(key)
            sizes.append(size)
        except OSError:
            continue
        except Exception:
            logger.exception("FSNative startup scan: failed to parse %s", path)
    return keys, sizes


class FSNativeL2AdapterConfig'''
    if old not in text:
        raise SystemExit('fs_native helper insertion point not found')
    text = text.replace(old, new, 1)

    old = '''    return NativeConnectorL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="FSNativeL2Adapter",
        extra_status={
            "base_path": config.base_path,
            "use_odirect": config.use_odirect,
            "num_workers": config.num_workers,
            "read_ahead_size": config.read_ahead_size,
        },
    )
'''
    new = '''    adapter = NativeConnectorL2Adapter(
        native_client,
        max_capacity_gb=config.max_capacity_gb,
        type_name="FSNativeL2Adapter",
        extra_status={
            "base_path": config.base_path,
            "use_odirect": config.use_odirect,
            "num_workers": config.num_workers,
            "read_ahead_size": config.read_ahead_size,
        },
    )
    try:
        keys, sizes = _scan_existing_fs_native_files(config.base_path)
        adapter.prime_existing_keys(keys, sizes)
    except Exception:
        logger.exception("FSNative startup scan failed")
    return adapter
'''
    if old not in text:
        raise SystemExit('fs_native factory return block not found')
    text = text.replace(old, new, 1)

    backup = FS_NATIVE_PATH.with_suffix('.py.bak_fs_native_scan')
    if not backup.exists():
        backup.write_text(FS_NATIVE_PATH.read_text())
    FS_NATIVE_PATH.write_text(text)
    print('patch_fs_native_startup_scan: patched fs_native factory')
    return True


if __name__ == '__main__':
    changed = patch_native_connector()
    changed = patch_fs_native_factory() or changed
    if changed:
        import py_compile
        py_compile.compile(str(NATIVE_PATH), doraise=True)
        py_compile.compile(str(FS_NATIVE_PATH), doraise=True)
        print('patch_fs_native_startup_scan: py_compile OK')
