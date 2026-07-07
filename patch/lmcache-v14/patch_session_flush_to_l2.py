#!/usr/bin/env python3
"""
Patch LMCache: flush stale L1 chunks to L2 without deleting them from L1.

Problem: When L1 is below the eviction watermark (which is the common case
with 32GB L1), chunks are never flushed to L2. If a session expires or the
container restarts, L1 entries are lost with no L2 backup → cold prefill.

The old patch (patch_ttl_flush_to_l2.py, removed 2026-07-02) scanned ALL
evictable keys every second and flushed+deleted 64/sec, draining L1 in
insertion order ("324-object ceiling" incident).

This patch takes a different approach:
  1. Runs every 30 seconds (not every 1 second)
  2. Flushes evictable L1 keys to L2 via store_objects_sync()
  3. Does NOT delete keys from L1 — they stay for fast L1 access
  4. store_objects_sync() already skips keys that exist in L2, so no
     redundant writes
  5. Rate-limited to 128 keys per flush cycle

Applied to: lmcache/v1/distributed/storage_controllers/eviction_controller.py
"""

import re
import sys

PATCH_MARKER = "# [PATCH: SESSION-FLUSH-v2]"

def apply_patch(content: str) -> str:
    if PATCH_MARKER in content:
        print("Already patched, skipping")
        return content

    # Find the eviction_loop method and add the periodic flush pass
    old_loop = '''    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                logger.debug(
                    "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage, watermark)
                self._publish_skipped(usage, watermark)
                continue

            logger.info(
                "L1 memory usage %.2f above watermark %.2f; triggering eviction.",
                usage, watermark)
            actions = self._eviction_policy.get_eviction_actions(
                eviction_ratio,
                key_eligible_filter=self._l1_manager.is_key_evictable,
            )
            for action in actions:
                self.execute_eviction_action(action)
            self._publish_triggered(usage, watermark)'''

    new_loop = '''    def eviction_loop(self):
        watermark = self._eviction_config.trigger_watermark
        eviction_ratio = self._eviction_config.eviction_ratio
        # [PATCH: SESSION-FLUSH-v2] Periodic background flush of evictable
        # L1 keys to L2 WITHOUT deleting them from L1. This ensures L2
        # always has a backup copy so that session expiry or container
        # restart doesn't cause a full cold prefill.
        _flush_counter = 0
        _FLUSH_INTERVAL = 10  # seconds between flush scans
        _FLUSH_BATCH = 256    # max keys per flush cycle

        while not self._stop_flag.is_set():
            time.sleep(1)
            used_bytes, total_bytes = self._l1_manager.get_memory_usage()
            usage = 0 if total_bytes == 0 else used_bytes / total_bytes
            if usage < watermark:
                # [PATCH: SESSION-FLUSH-v2] Periodic L2 backup flush.
                # Runs every _FLUSH_INTERVAL seconds when below watermark.
                # Copies evictable L1 keys to L2 but does NOT delete them
                # from L1, preserving hot keys for fast L1 access.
                _flush_counter += 1
                if _flush_counter >= _FLUSH_INTERVAL and self._l2_adapters:
                    _flush_counter = 0
                    self._backup_to_l2_no_delete(_FLUSH_BATCH)
                logger.debug(
                    "L1 memory usage %.2f below watermark %.2f; skipping eviction.",
                    usage, watermark)
                self._publish_skipped(usage, watermark)
                continue

            logger.info(
                "L1 memory usage %.2f above watermark %.2f; triggering eviction.",
                usage, watermark)
            actions = self._eviction_policy.get_eviction_actions(
                eviction_ratio,
                key_eligible_filter=self._l1_manager.is_key_evictable,
            )
            for action in actions:
                self.execute_eviction_action(action)
            self._publish_triggered(usage, watermark)

    def _backup_to_l2_no_delete(self, batch_limit: int) -> None:
        """[PATCH: SESSION-FLUSH-v2] Flush evictable L1 keys to L2 without
        deleting them from L1.

        This is a BACKUP operation, not an eviction. Keys remain in L1 for
        fast access. L2 gets a copy so that if L1 entries are lost (session
        expiry, container restart), the next request can restore from L2.

        store_objects_sync() skips keys that already exist in L2, so this
        is idempotent and doesn't do redundant disk writes.
        """
        if not self._l2_adapters:
            return

        # Collect evictable keys (not locked by any active request)
        evictable_keys = []
        for key in list(self._l1_manager._objects.keys()):
            if self._l1_manager.is_key_evictable(key):
                evictable_keys.append(key)

        if not evictable_keys:
            return

        # Limit batch size to avoid blocking the eviction loop, but rotate a
        # cursor so repeated scans cover the whole L1 keyspace instead of
        # hammering the first N insertion-ordered keys forever.
        start = getattr(self, "_session_flush_cursor", 0) % len(evictable_keys)
        ordered = evictable_keys[start:] + evictable_keys[:start]
        batch = ordered[:batch_limit]
        self._session_flush_cursor = (start + len(batch)) % len(evictable_keys)

        # Read L1 objects (reserve_read → finish_read, no delete)
        read_result = self._l1_manager.reserve_read(batch)
        readable_keys = []
        readable_objs = []
        for key in batch:
            entry = read_result.get(key)
            if entry is not None and entry[0] == L1Error.SUCCESS and entry[1] is not None:
                readable_keys.append(key)
                readable_objs.append(entry[1])

        if not readable_keys:
            return

        # Flush to L2 (store_objects_sync skips existing keys)
        total_persisted = 0
        total_bytes = 0
        adapters = (list(self._l2_adapters.values())
                    if isinstance(self._l2_adapters, dict)
                    else self._l2_adapters)
        for adapter in adapters:
            sync_store = getattr(adapter, "store_objects_sync", None)
            if sync_store is None:
                continue
            try:
                ok, persisted, written = sync_store(readable_keys, readable_objs)
                if ok:
                    total_persisted += persisted
                    total_bytes += written
            except Exception:
                logger.exception("SESSION-FLUSH: sync store failed")

        # Release read locks (but do NOT delete from L1)
        self._l1_manager.finish_read(readable_keys)

        if total_bytes > 0:
            logger.info(
                "SESSION-FLUSH: backed up %d keys to L2 (%d new bytes, "
                "%d evictable remaining in L1, not deleted)",
                total_persisted, total_bytes, len(evictable_keys) - len(batch))
        elif total_persisted > 0:
            logger.debug(
                "SESSION-FLUSH: checked %d keys; all already present in L2 "
                "(%d evictable remaining in L1, not deleted)",
                total_persisted, len(evictable_keys) - len(batch))'''

    if old_loop not in content:
        print("ERROR: Could not find eviction_loop to patch")
        # Try to find a partial match for debugging
        if "def eviction_loop" in content:
            start = content.index("def eviction_loop")
            print("Found 'def eviction_loop' at offset", start)
            print("Context:", content[start:start+200])
        sys.exit(1)

    content = content.replace(old_loop, new_loop)
    print("Patch applied successfully")
    return content


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/opt/venv/lib/python3.12/site-packages/lmcache/v1/distributed/storage_controllers/eviction_controller.py"
    with open(path, "r") as f:
        content = f.read()
    patched = apply_patch(content)
    with open(path, "w") as f:
        f.write(patched)
    print(f"Written to {path}")
