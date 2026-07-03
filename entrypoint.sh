#!/usr/bin/env bash
# Thin wrapper around the image's serve-glm52.sh.
# The image bakes NCCL_GRAPH_FILE="" (empty), which makes this "-noxml" NCCL
# build try to open an XML topology graph at path "" and abort ncclCommInitRank
# with "unhandled system error". Unsetting it (and disabling XML topo, matching
# the known-good Kimi-v5 config on this host) fixes multi-GPU init.
unset NCCL_GRAPH_FILE
export USE_NCCL_XML=0
# 2026-07-03: GPU P2P WORKS on this host now. Root causes fixed: (1) forced-P2P
# driver registry keys (ForceP2P/RMForceP2PType/RMPcieP2PType) made the driver
# advertise P2P that failed to map ("invalid device ordinal") — removed from
# /etc/modprobe.d/nvidia-p2p-override.conf on the host + module reload; (2) ACS
# ReqRedir/CmpltRedir on the PEX890xx switch ports hairpinned P2P through the
# root complex — cleared via setpci (runtime; re-apply after host reboot!).
# Measured: 54 GB/s intra-socket, 49 GB/s cross-socket, prefill +50%, decode +16-24%.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
# Optional NUMA memory interleave (NUMA_INTERLEAVE=1): on this 2-socket host the
# NCCL SHM staging buffers live in host RAM; interleaving spreads them across both
# nodes instead of wherever the allocating worker happens to sit.
if [[ "${NUMA_INTERLEAVE:-0}" == "1" ]] && command -v numactl >/dev/null; then
  exec numactl --interleave=all /usr/local/bin/serve-glm52.sh "$@"
fi
exec /usr/local/bin/serve-glm52.sh "$@"
