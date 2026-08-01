# Day 8 Nsight Compute 报告索引

## 固定配置

```text
GPU: NVIDIA GeForce RTX 2080 Ti
workload: 4096×4096 float transpose
grid: (128,128,1)
block: (32,8,1)
build: RelWithDebInfo + lineinfo
```

## 报告

| 文件 | 目标 | 主要用途 |
|---|---|---|
| `day08_before.ncu-rep` | `transpose_tiled_kernel` | Speed of Light、Memory、Occupancy、Warp State、Source |
| `day08_after.ncu-rep` | `transpose_padded_kernel` | 与 before 使用相同采集配置对比 |
| `day08_before_bank.ncu-rep` | tiled 自定义 bank metrics | shared load/store conflict 与 wavefront |
| `day08_after_bank.ncu-rep` | padded 自定义 bank metrics | 验证 padding 后 conflict 为 0 |

## 核心 before/after

| 指标 | before | after |
|---|---:|---:|
| Duration | 373.95 μs | 280.16 μs |
| L1/TEX Throughput | 96.36% | 16.89% |
| DRAM Throughput | 55.42% | 72.48% |
| Achieved Occupancy | 92.71% | 97.63% |
| Warp Cycles/Issued Instruction | 124.14 | 98.04 |
| Shared-load conflicts | 16,252,928 | 0 |
| Shared-store conflicts | 0 | 0 |
| Shared-load wavefronts | 16,777,216（推导） | 524,288 |

## 结论

`tile[32][32]` 的转置读取造成 32-way shared-memory bank conflict。`tile[32][33]` padding 将冲突归零，使 MIO/L1TEX 压力下降；普通 benchmark 中 `4096×4096` padded 相对 tiled 获得约 `1.307×` 加速。

完整分析：[Day 8 总览](../doc/day08/README.md) 与 [专题分析](../doc/day08/day08_ncu_transpose_analysis.md)。
