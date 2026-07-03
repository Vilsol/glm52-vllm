#!/usr/bin/env bash
###############################################################################
# p2p-module-reload.sh — reload the NVIDIA kernel module with honest P2P probing.
#
# Context (2026-07-03, node dev-k8s-rtx02): the driver was loaded with forced-P2P
# registry keys (ForceP2P/RMForceP2PType/RMPcieP2PType) that make it ADVERTISE
# P2P which then fails to map ("invalid device ordinal"). The modprobe conf has
# already been cleaned (/etc/modprobe.d/nvidia-p2p-override.conf on host) and the
# PCIe switch ACS redirect bits cleared (runtime). This script completes the fix:
# evict GPU-holding pods -> stop persistenced -> unload/reload module -> verify ->
# probe P2P -> restore everything.
#
# Run from the workspace pod (needs: hostPID, privileged, /host mount, kubectl
# RBAC for node labels + pod list/delete). Idempotent-ish; safe to re-run after
# a partial failure — every phase checks state before acting.
#
# Rollback: old modprobe conf at grid_results/nvidia-p2p-override.conf.bak;
# `bash p2p-module-reload.sh --restore-only` re-enables pods/persistenced only.
###############################################################################
set -uo pipefail
export PATH=/root/bin:$PATH

NODE=dev-k8s-rtx02
GLM=/root/glm52-vllm
OUT=$GLM/grid_results
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v8722ac7-b12x8ce61f9-cu132-20260629
# gpu-operator per-node deploy labels for every GPU-holding daemonset on this node
DEPLOY_LABELS=(dcgm-exporter device-plugin gpu-feature-discovery mig-manager operator-validator)
ACS_PORTS=(02:00.0 02:01.0 60:00.0 60:01.0 88:00.0 88:01.0 da:00.0 da:01.0)
CLEAN_KEYS='GrdmaPciTopoCheckOverride=1;EnableResizableBar=1'

log()  { echo "[$(date '+%T')] $*"; }
host() { nsenter -t 1 -m -u -i -n -p -- "$@"; }
uvm_refs() { awk '$1=="nvidia_uvm"{print $3}' /proc/modules; }

restore() {
  log "RESTORE: re-enabling GPU daemonset pods + persistenced"
  labels=""; for l in "${DEPLOY_LABELS[@]}"; do labels+=" nvidia.com/gpu.deploy.$l=true"; done
  kubectl label node "$NODE" $labels --overwrite
  host systemctl start nvidia-persistenced 2>/dev/null || true
  host nvidia-smi -pm 1 >/dev/null 2>&1 || true
  for i in $(seq 1 24); do
    n=$(kubectl get pods -n gpu-system -o wide --field-selector spec.nodeName=$NODE 2>/dev/null \
        | grep -cE "dcgm-exporter|device-plugin" )
    [ "$n" -ge 2 ] && { log "RESTORE: gpu-system pods back ($n)"; return 0; }
    sleep 5
  done
  log "RESTORE WARNING: gpu-system pods not confirmed back after 120s — check manually"
}
[ "${1:-}" = "--restore-only" ] && { restore; exit 0; }

fail() { log "FATAL: $*"; restore; exit 1; }

# ---------- phase 0: preflight ----------
log "phase 0: preflight"
kubectl auth can-i patch nodes >/dev/null      || fail "kubectl RBAC missing"
host true                                       || fail "nsenter to host failed"
grep -q "$CLEAN_KEYS" /host/etc/modprobe.d/nvidia-p2p-override.conf \
                                                || fail "modprobe conf not cleaned"
docker rm -f glm52 >/dev/null 2>&1 || true      # ensure no inference server holds GPUs
for p in "${ACS_PORTS[@]}"; do
  v=$(setpci -s "$p" ECAP_ACS+6.w)
  [ "$v" = "0001" ] || { log "  ACS $p=$v -> clearing"; setpci -s "$p" ECAP_ACS+6.w=0x0001; }
done
log "  ACS redirect clear on all 8 ports; uvm refs now: $(uvm_refs)"

# ---------- phase 1: evict GPU pods ----------
log "phase 1: evicting GPU daemonset pods via deploy labels"
labels=""; for l in "${DEPLOY_LABELS[@]}"; do labels+=" nvidia.com/gpu.deploy.$l=false"; done
kubectl label node "$NODE" $labels --overwrite
for i in $(seq 1 36); do
  left=$(kubectl get pods -n gpu-system -o wide --field-selector spec.nodeName=$NODE 2>/dev/null \
         | grep -E "dcgm|device-plugin|gpu-feature-discovery|mig-manager|operator-validator" \
         | grep -v "node-feature-discovery" | grep -cv Completed)
  [ "$left" -eq 0 ] && break
  sleep 5
done
[ "${left:-1}" -eq 0 ] || fail "GPU pods still on node after 180s"
log "  GPU pods evicted"

# ---------- phase 2: quiesce host driver clients ----------
log "phase 2: stopping persistenced + persistence mode"
host systemctl stop nvidia-persistenced 2>/dev/null || true
host nvidia-smi -pm 0 >/dev/null 2>&1 || true
for i in $(seq 1 24); do
  r=$(uvm_refs); [ "${r:-1}" = "0" ] && break; sleep 5
done
if [ "$(uvm_refs)" != "0" ]; then
  log "  holders still present; visible GPU-fd processes:"
  for p in /proc/[0-9]*; do
    ls -l "$p/fd" 2>/dev/null | grep -q "/dev/nvidia" && echo "    pid ${p#/proc/} $(cat $p/comm 2>/dev/null)"
  done
  fail "nvidia_uvm refs stuck at $(uvm_refs) after 120s"
fi
log "  uvm refs = 0"

# ---------- phase 3: reload module ----------
log "phase 3: unloading nvidia modules"
for m in nvidia_drm nvidia_modeset nvidia_uvm nvidia; do
  grep -q "^${m} " /proc/modules && { host rmmod "$m" || fail "rmmod $m failed"; }
done
grep -q "^nvidia " /proc/modules && fail "nvidia still loaded after rmmod"
log "  unloaded. reloading via host modprobe (reads cleaned modprobe.d)"
host modprobe nvidia        || fail "modprobe nvidia failed — RUN NOTHING; tell user (reboot path)"
host modprobe nvidia_uvm    || fail "modprobe nvidia_uvm failed"
host modprobe nvidia_modeset 2>/dev/null || true   # parity with pre-experiment state
host modprobe nvidia_drm     2>/dev/null || true

# ---------- phase 4: verify ----------
log "phase 4: verify driver state"
params=$(grep RegistryDwords: /proc/driver/nvidia/params)
log "  $params"
echo "$params" | grep -q "ForceP2P" && fail "force keys still present after reload?!"
host nvidia-smi -L | tee "$OUT/p2p_reload_gpus.txt" | head -2
n=$(host nvidia-smi -L | grep -c "^GPU") ; [ "$n" = "8" ] || fail "only $n/8 GPUs visible"
host nvidia-smi -pm 1 >/dev/null 2>&1 || true
for p in "${ACS_PORTS[@]}"; do
  [ "$(setpci -s $p ECAP_ACS+6.w)" = "0001" ] || fail "ACS reverted on $p"
done
log "  8 GPUs up, clean params, ACS still clear"

# ---------- phase 5: P2P probes ----------
log "phase 5: P2P probes (results -> $OUT/p2p_after_reload.*)"
docker run --rm --privileged --device nvidia.com/gpu=all --entrypoint /opt/venv/bin/python \
  -v /root/.claude/jobs/8d077c7c/tmp/p2p_probe.py:/p2p_probe.py:ro "$IMAGE" /p2p_probe.py 2>&1 \
  | grep -E "^GPU" | tee "$OUT/p2p_after_reload.torch.txt"
docker run --rm --privileged --device nvidia.com/gpu=all --entrypoint bash \
  -v /root/p2pmark:/p2pmark:ro "$IMAGE" -c 'cd /p2pmark && ./p2pmark 2>&1' \
  | head -22 | tee "$OUT/p2p_after_reload.p2pmark.txt"

# ---------- phase 6: restore cluster state ----------
restore
log "DONE. Compare $OUT/p2p_after_reload.p2pmark.txt vs $OUT/acs_baseline_p2pmark.txt"
