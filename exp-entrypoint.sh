#!/usr/bin/env bash
unset NCCL_GRAPH_FILE
export USE_NCCL_XML=0 NCCL_P2P_DISABLE=1 PYTHON_BIN=/opt/venv/bin/python
exec /exp-serve.sh "$@"
