#!/usr/bin/env bash
# glm52 health watchdog — restart the vLLM container if the engine dies while the
# container stays "Up" (the 2026-07-16 B12X-indexer failure mode: EngineCore dead,
# API serving 500s, docker restart policy never triggers because the process lives).
#
# Boot-aware: never restarts during a legitimate (re)boot — it only force-acts on a
# hung boot after BOOT_GRACE. Once it has seen the server healthy, a subsequent health
# failure is treated as a crash and restarted quickly.
# Backoff: at most MAX_RESTARTS within RESTART_WINDOW, then it STOPS auto-restarting and
# logs loudly (so a poisoned request can't cause an infinite cold-restart loop).
set -uo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

NAME=glm52
URL=http://127.0.0.1:8443/health
LAUNCH="bash /root/glm52-vllm/vllm-v19-lmcache.sh"   # current production (v19/DCP2/LMCache/CKV)
LOG=/root/glm52-vllm/watchdog.log

CHECK_INTERVAL=15     # seconds between health probes
FAIL_THRESHOLD=4      # consecutive failures (~60s) before acting
BOOT_GRACE=1800       # after a (re)start, allow this long to boot before force-restart
MAX_RESTARTS=3        # max restarts within RESTART_WINDOW before giving up
RESTART_WINDOW=3600

log(){ echo "$(date -u +%FT%TZ) $*" >>"$LOG"; }

healthy=0                    # have we seen HTTP 200 since the last (re)start?
fails=0
last_start=$(date +%s)       # watchdog assumes the server is mid-life when it starts
restart_times=""             # space-separated epochs of recent restarts

log "watchdog started (pid $$) — url=$URL interval=${CHECK_INTERVAL}s threshold=${FAIL_THRESHOLD} max_restarts=${MAX_RESTARTS}/${RESTART_WINDOW}s"

while true; do
  now=$(date +%s)
  code=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null || echo 000)

  if [ "$code" = "200" ]; then
    [ "$healthy" = 0 ] && log "healthy (HTTP 200)"
    healthy=1; fails=0
    sleep "$CHECK_INTERVAL"; continue
  fi

  fails=$((fails+1))

  should=0
  if [ "$healthy" = 1 ] && [ "$fails" -ge "$FAIL_THRESHOLD" ]; then
    should=1; reason="engine died after being healthy (HTTP $code x$fails)"
  elif [ "$healthy" = 0 ] && [ $((now-last_start)) -ge "$BOOT_GRACE" ] && [ "$fails" -ge "$FAIL_THRESHOLD" ]; then
    should=1; reason="boot did not become healthy within ${BOOT_GRACE}s (HTTP $code)"
  fi

  if [ "$should" = 1 ]; then
    # prune restart history outside the window
    pruned=""; for t in $restart_times; do [ $((now-t)) -lt "$RESTART_WINDOW" ] && pruned="$pruned $t"; done
    restart_times="$pruned"
    count=$(echo $restart_times | wc -w)
    if [ "$count" -ge "$MAX_RESTARTS" ]; then
      log "!!! UNHEALTHY: $reason — but ${count} restarts already in the last ${RESTART_WINDOW}s. GIVING UP auto-restart; MANUAL INTERVENTION NEEDED."
      sleep 300; continue
    fi
    log ">>> UNHEALTHY: $reason — restarting '$NAME' (restart $((count+1))/${MAX_RESTARTS} in window)"
    $LAUNCH >>"$LOG" 2>&1
    restart_times="$restart_times $(date +%s)"
    last_start=$(date +%s); healthy=0; fails=0
    log ">>> restart issued; boot grace ${BOOT_GRACE}s begins"
  fi

  sleep "$CHECK_INTERVAL"
done
