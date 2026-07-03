# Host hardware — 8x RTX PRO 6000 Blackwell, dual EPYC (anonymized)

Spec sheet for the machine behind `BENCHMARKS.md`. Each fact lists the command
that produced it (2026-07-03, post Phase-10 P2P fix).

## Summary

| Component | Value | Source |
|---|---|---|
| CPU | 2x AMD EPYC 9375F (Zen5, 32c/64t each; 128 threads) | `grep "model name" /proc/cpuinfo`, `nproc` |
| RAM | 2 TB (1 TB/NUMA node), no hugepages | `numactl --hardware`, `/proc/meminfo` |
| NUMA | 2 nodes; distance 10 local / 32 remote; node0=cpus 0-31,64-95; node1=32-63,96-127 | `numactl --hardware` |
| GPUs | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB (97887 MiB), no NVLink | `nvidia-smi --query-gpu=...` |
| GPU driver | 580.119.02 open kernel module (distro-packaged); `NVreg_RegistryDwords="GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"` | `/proc/driver/nvidia/version`, `/proc/driver/nvidia/params` |
| PCIe switches | 4x Broadcom PEX89000 Gen5 (`1000:c030` rev b0), 2 GPUs each + 3 empty ports + mgmt | `lspci -nn \| grep PEX` |
| Links | Gen5 x16 (32 GT/s) everywhere, GPU and uplink | `lspci -vv -s <port> \| grep Lnk` |
| IOMMU | AMD-Vi active (8 IVHD units), translation mode (`iommu=pt` NOT set) | `ls /sys/class/iommu`, `/proc/cmdline` |
| OS / kernel | openSUSE Leap Micro 6.1 (k8s node), 6.4.0-39-default | `/etc/os-release`, `uname -r` |
| Storage | `/root` on NFS PVC; ~460 GB weights page-cached in RAM when warm | `df /root` |

## Topology

```
   NUMA 0 (EPYC #1, 1TB RAM)                  NUMA 1 (EPYC #2, 1TB RAM)
   ┌────────────────────────┐    xGMI     ┌────────────────────────┐
   │        Socket 0        │◄══════════►│        Socket 1        │
   └───┬───────────────┬────┘  ~155GB/s  └───┬───────────────┬────┘
       │ Gen5 x16      │ Gen5 x16            │ Gen5 x16      │ Gen5 x16
   ┌───▼────┐      ┌───▼────┐            ┌───▼────┐      ┌───▼────┐
   │ PEX89k │      │ PEX89k │            │ PEX89k │      │ PEX89k │
   │  sw A  │      │  sw B  │            │  sw C  │      │  sw D  │
   │(bus 02)│      │(bus 60)│            │(bus 88)│      │(bus da)│
   └┬─┬─┬───┘      └┬─┬─┬───┘            └┬─┬─┬───┘      └┬─┬─┬───┘
    │ │ └3 empty    │ │ └3 empty          │ │ └3 empty    │ │ └3 empty
  ┌─▼┐┌▼─┐        ┌─▼┐┌▼─┐              ┌─▼┐┌▼─┐        ┌─▼┐┌▼─┐
  │G0││G1│        │G2││G3│              │G4││G5│        │G6││G7│
  │03││0b│        │61││69│              │89││91│        │db││e3│
  └──┘└──┘        └──┘└──┘              └──┘└──┘        └──┘└──┘
```

| switch | upstream | GPU ports | GPUs |
|---|---|---|---|
| A (NUMA0) | 01:00.0 | 02:00.0, 02:01.0 | 03:00.0, 0b:00.0 |
| B (NUMA0) | 5f:00.0 | 60:00.0, 60:01.0 | 61:00.0, 69:00.0 |
| C (NUMA1) | 87:00.0 | 88:00.0, 88:01.0 | 89:00.0, 91:00.0 |
| D (NUMA1) | d9:00.0 | da:00.0, da:01.0 | db:00.0, e3:00.0 |

`nvidia-smi topo -m`: PIX pairs (0,1)(2,3)(4,5)(6,7); NODE within a socket;
SYS across sockets.

## P2P state (fixed 2026-07-03, BENCHMARKS.md Phase 10)

- Driver keys `ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2` removed from
  host `/etc/modprobe.d/nvidia-p2p-override.conf` (they broke all P2P mappings:
  `CUDA invalid device ordinal`).
- ACS on the 8 GPU switch ports: `0x001d -> 0x0001`
  (`setpci -s <port> ECAP_ACS+6.w=0x0001`; cap at cfg offset 0x170, ctl 0x176).
  Persisted by host systemd unit `gpu-acs-p2p.service` (dd to config space,
  `Before=kubelet.service`).
- Serving config: `NCCL_P2P_DISABLE=0`, `VLLM_ENABLE_PCIE_ALLREDUCE=0`.

## P2P bandwidth, before/after (`p2pmark`, 152 MB x 20 iters)

BEFORE — ACS redirect + broken driver P2P; everything host-staged, flat ~42,
same-switch pairs slowest (uplink contention):

```
 Dst->  GPU0     GPU1     GPU2     GPU3     GPU4     GPU5     GPU6     GPU7
GPU 0:     -      40.64     42.23     42.33     42.90     41.94     43.01     42.78
GPU 1:   40.62       -      42.34     42.35     43.01     42.13     42.99     42.86
GPU 2:   42.28     42.35       -      40.55     42.77     41.97     43.06     42.70
GPU 3:   42.37     42.32     40.36       -      43.08     42.14     43.21     42.93
GPU 4:   42.31     42.40     42.31     42.50       -      40.63     43.28     43.10
GPU 5:   42.24     42.20     42.25     42.44     41.04       -      42.94     42.77
GPU 6:   42.36     42.32     42.30     41.89     43.20     42.19       -      41.20
GPU 7:   42.31     42.39     42.39     42.42     43.20     42.15     41.38       -
```

AFTER — direct P2P; ~54 intra-socket, ~49 cross-socket:

```
 Dst->  GPU0     GPU1     GPU2     GPU3     GPU4     GPU5     GPU6     GPU7
GPU 0:     -      54.08     54.15     54.20     49.79     49.86     49.57     50.26
GPU 1:   54.42       -      54.31     54.74     49.69     48.95     49.16     49.81
GPU 2:   54.33     53.57       -      54.53     48.70     49.16     48.93     49.86
GPU 3:   53.83     53.52     53.66       -      49.04     48.89     48.85     48.82
GPU 4:   49.63     49.02     50.00     49.20       -      53.81     53.91     53.97
GPU 5:   49.46     49.95     48.86     49.64     54.13       -      54.42     54.45
GPU 6:   49.31     49.02     49.50     49.05     53.72     53.41       -      54.29
GPU 7:   49.62     49.42     49.28     49.75     54.09     53.91     53.44       -
```

Concurrent load (all 8 GPUs, fixed peer distance): intra-socket rounds ~383-389
GB/s total; all-cross-socket round 155.89 GB/s total (~19.5/pair) = xGMI ceiling.

Raw outputs (local only, `grid_results/` not committed):
`acs_baseline_p2pmark.txt`, `p2p_after_reload.p2pmark.txt`.
