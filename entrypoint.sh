#!/usr/bin/env bash
# Thin wrapper around the image's serve-glm52.sh.
# The image bakes NCCL_GRAPH_FILE="" (empty), which makes this "-noxml" NCCL
# build try to open an XML topology graph at path "" and abort ncclCommInitRank
# with "unhandled system error". Unsetting it (and disabling XML topo, matching
# the known-good Kimi-v5 config on this host) fixes multi-GPU init.
unset NCCL_GRAPH_FILE
export USE_NCCL_XML=0
# These RTX PRO 6000 cards are PCIe-only across two NUMA nodes; NCCL's direct P2P
# transport fails here ("Cuda failure 101 'invalid device ordinal'"). Force NCCL
# onto its SHM transport. The b12x custom PCIe all-reduce still handles the TP
# hot path, so throughput is unaffected. (Matches the known-good Kimi-v5 config.)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
exec /usr/local/bin/serve-glm52.sh "$@"
