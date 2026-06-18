#!/usr/bin/env bash
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS 2>/dev/null || true
export USE_NCCL_XML=0 NCCL_P2P_DISABLE=1 PYTHON_BIN=/opt/venv/bin/python
export NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL,LL128,Simple

# The repo serve.sh omits the b12x env that serve-glm52.sh exports. Without it,
# vLLM uses its STANDARD custom all-reduce, which needs CUDA IPC peer access
# (blocked in this pod) -> "invalid device ordinal". Exporting the b12x set
# routes all-reduce through the b12x PCIe path (graceful NCCL fallback here) and
# enables the b12x MoE/sparse/AOT kernels.
export VLLM_USE_AOT_COMPILE=1 VLLM_USE_MEGA_AOT_ARTIFACT=1 VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_B12X_FP8_GEMM=1 VLLM_USE_B12X_MOE=1 VLLM_USE_B12X_SPARSE_INDEXER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
export B12X_DENSE_SPLITK_TURBO=1 B12X_W4A16_TC_DECODE=1

# serve.sh expects the DCP patches at /opt/patch_*.py
cp /lmcache-mnt/patch_*.py /opt/ 2>/dev/null || true
exec bash /lmcache-mnt/serve.sh
