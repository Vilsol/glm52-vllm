#!/usr/bin/env python3
"""Patch NativeConnectorL2Adapter: add race-free synchronous store_objects_sync().

Why:
- We switched L2 from Python fs adapter to fs_native for parallel C++ I/O.
- Existing L1 eviction flush patch calls adapter.store_objects_sync().
- FSL2Adapter has that method, but NativeConnectorL2Adapter did not.
- Without it, L1 eviction under memory pressure preserves readable keys instead
  of deleting them after flush, so L1 can remain over watermark.

This patch adds a dedicated sync-store op type routed by the demux thread to a
private Event/result slot instead of the shared pop_completed_store_tasks()
queue. That avoids the StoreController race that the earlier FSL2 sync patch
was designed to eliminate.
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

TARGET = Path("/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py")
MARKER = "[NATIVE-SYNC-STORE v1]"


def fail(msg: str) -> None:
    print(f"❌ {msg}")
    sys.exit(1)


def backup(path: Path, suffix: str) -> None:
    b = path.with_name(path.name + suffix)
    if not b.exists() and path.exists():
        b.write_text(path.read_text(errors="replace"))
        print(f"✅ backup {b}")


def compile_file(path: Path) -> None:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        fail(f"syntax check failed {path}: {e}")


SYNC_METHOD = f'''    def store_objects_sync(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> tuple[bool, int, int]:
        """Synchronously persist objects to native L2. {MARKER}

        Used by L1 eviction. This deliberately does NOT use the shared
        ``pop_completed_store_tasks()`` queue, so StoreController cannot race
        with the eviction thread for completions.

        Returns:
            (success, persisted_count, bytes_written). persisted_count counts
            backend-accepted keys; bytes_written counts bytes newly accounted
            in this adapter process.
        """
        if not keys:
            return True, 0, 0

        key_strings = [_object_key_to_string(k) for k in keys]
        memviews = [_obj_to_memoryview(obj) for obj in objects]
        per_key_sizes = [obj.get_size() for obj in objects]
        done_event = threading.Event()
        result: dict[str, Any] = {{
            "ok": False,
            "persisted_count": 0,
            "bytes_written": 0,
        }}

        with self._lock:
            task_id = self._get_next_task_id()
            future_id = int(self._client.submit_batch_set(key_strings, memviews))
            self._pending_ops[future_id] = (
                self._OP_SYNC_STORE,
                task_id,
                len(keys),
                None,
            )
            self._pending_store_sizes[future_id] = (list(keys), per_key_sizes)
            self._pending_sync_store_events[task_id] = (done_event, result)

        timeout = max(30.0, min(300.0, len(keys) * 5.0))
        if not done_event.wait(timeout=timeout):
            with self._lock:
                self._pending_sync_store_events.pop(task_id, None)
                for fid, entry in list(self._pending_ops.items()):
                    if entry[1] == task_id:
                        self._pending_ops.pop(fid, None)
                        self._pending_store_sizes.pop(fid, None)
                        break
            logger.warning(
                "store_objects_sync() timed out after %.1fs for %d keys",
                timeout,
                len(keys),
            )
            return False, 0, 0

        return (
            bool(result.get("ok", False)),
            int(result.get("persisted_count", 0)),
            int(result.get("bytes_written", 0)),
        )

'''


def main() -> None:
    if not TARGET.exists():
        fail(f"missing target {TARGET}")

    text = TARGET.read_text()
    if MARKER in text:
        print("↪️ native_connector_l2_adapter.py: native sync store already applied")
        return

    backup(TARGET, ".bak_native_sync_store_v1")

    replacements: list[tuple[str, str, str]] = [
        (
            'op tag',
            '    _OP_STORE = "store"\n    _OP_LOOKUP = "lookup"\n    _OP_LOAD = "load"\n    _OP_DELETE = "delete"\n',
            '    _OP_STORE = "store"\n    _OP_SYNC_STORE = "sync_store"\n    _OP_LOOKUP = "lookup"\n    _OP_LOAD = "load"\n    _OP_DELETE = "delete"\n',
        ),
        (
            'sync event state',
            '        # Pending delete events for synchronous delete() calls\n        self._pending_delete_events: dict[L2TaskId, threading.Event] = {}\n\n        # Per-key size tracking.',
            '        # Pending delete events for synchronous delete() calls\n        self._pending_delete_events: dict[L2TaskId, threading.Event] = {}\n\n        # Pending native sync-store events for L1 eviction. These are kept\n        # separate from _completed_stores so StoreController cannot consume\n        # eviction-thread completions.\n        self._pending_sync_store_events: dict[\n            L2TaskId, tuple[threading.Event, dict[str, Any]]\n        ] = {}\n\n        # Per-key size tracking.',
        ),
        (
            'sync method',
            '    def pop_completed_store_tasks(\n        self,\n    ) -> dict[L2TaskId, L2StoreResult]:\n        with self._lock:\n            completed = self._completed_stores\n            self._completed_stores = {}\n        return completed\n\n    # ---------------------------------------------------------------\n    # Lookup and Lock Interface\n',
            '    def pop_completed_store_tasks(\n        self,\n    ) -> dict[L2TaskId, L2StoreResult]:\n        with self._lock:\n            completed = self._completed_stores\n            self._completed_stores = {}\n        return completed\n\n' + SYNC_METHOD + '    # ---------------------------------------------------------------\n    # Lookup and Lock Interface\n',
        ),
        (
            'demux local sync events',
            '            delete_done_events: list[threading.Event] = []\n\n            with self._lock:',
            '            delete_done_events: list[threading.Event] = []\n            sync_store_done_events: list[threading.Event] = []\n\n            with self._lock:',
        ),
        (
            'store demux branch',
            '                    if op_type == self._OP_STORE:\n                        store_info = self._pending_store_sizes.pop(fid, None)\n                        task_bytes = 0\n                        if ok and store_info is not None:\n                            store_keys, sizes = store_info\n                            for key, size in zip(store_keys, sizes, strict=True):\n                                # First-store wins for byte accounting:\n                                # a re-store of an existing key adds 0\n                                # bytes (the backend already holds it).\n                                # We still notify the listener for every\n                                # store so LRU policies can ``move_to_end``\n                                # on re-store — passing size=0 in that\n                                # case is a no-op for the base counters.\n                                if key not in self._key_sizes:\n                                    self._key_sizes[key] = size\n                                    keys_stored.append(key)\n                                    sizes_stored.append(size)\n                                    task_bytes += size\n                                else:\n                                    keys_stored.append(key)\n                                    sizes_stored.append(0)\n                        self._completed_stores[task_id] = L2StoreResult(ok, task_bytes)\n                        self._store_efd.notify()\n\n                    elif op_type == self._OP_LOOKUP:',
            '                    if op_type == self._OP_STORE or op_type == self._OP_SYNC_STORE:\n                        store_info = self._pending_store_sizes.pop(fid, None)\n                        task_bytes = 0\n                        persisted_count = 0\n                        if ok and store_info is not None:\n                            store_keys, sizes = store_info\n                            persisted_count = len(store_keys)\n                            for key, size in zip(store_keys, sizes, strict=True):\n                                # First-store wins for byte accounting:\n                                # a re-store of an existing key adds 0\n                                # bytes (the backend already holds it).\n                                # We still notify the listener for every\n                                # store so LRU policies can ``move_to_end``\n                                # on re-store — passing size=0 in that\n                                # case is a no-op for the base counters.\n                                if key not in self._key_sizes:\n                                    self._key_sizes[key] = size\n                                    keys_stored.append(key)\n                                    sizes_stored.append(size)\n                                    task_bytes += size\n                                else:\n                                    keys_stored.append(key)\n                                    sizes_stored.append(0)\n\n                        if op_type == self._OP_STORE:\n                            self._completed_stores[task_id] = L2StoreResult(ok, task_bytes)\n                            self._store_efd.notify()\n                        else:\n                            sync_entry = self._pending_sync_store_events.pop(task_id, None)\n                            if sync_entry is not None:\n                                evt, result = sync_entry\n                                result["ok"] = bool(ok)\n                                result["persisted_count"] = persisted_count\n                                result["bytes_written"] = task_bytes\n                                sync_store_done_events.append(evt)\n\n                    elif op_type == self._OP_LOOKUP:',
        ),
        (
            'set sync events after notify',
            '            if keys_deleted:\n                self._notify_keys_deleted(keys_deleted, sizes_deleted)\n            # Unblock any synchronous ``delete()`` callers only AFTER\n',
            '            if keys_deleted:\n                self._notify_keys_deleted(keys_deleted, sizes_deleted)\n            # Unblock native sync-store callers only AFTER store notifications\n            # have updated base-class byte/accounting state.\n            for evt in sync_store_done_events:\n                evt.set()\n            # Unblock any synchronous ``delete()`` callers only AFTER\n',
        ),
    ]

    for name, old, new in replacements:
        if old not in text:
            fail(f"native_connector_l2_adapter.py: replacement marker not found: {name}")
        text = text.replace(old, new, 1)

    TARGET.write_text(text)
    compile_file(TARGET)

    verify = TARGET.read_text()
    for needle in [MARKER, "_OP_SYNC_STORE", "_pending_sync_store_events", "def store_objects_sync"]:
        if needle not in verify:
            fail(f"verification missing {needle}")
    print("✅ native_connector_l2_adapter.py: added native store_objects_sync v1")


if __name__ == "__main__":
    main()
