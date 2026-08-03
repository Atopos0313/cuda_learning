# Day 9 分析归档

## 状态

Day 9 已完成。正确性、Release benchmark、完成同步边界和 Nsight Systems 时间线证据均已归档。

## 报告索引

- [Day 9 学习总览](../doc/day09/README.md)
- [Nsight Systems 流水线专题分析](../doc/day09/day09_nsys_stream_pipeline_analysis.md)
- [C++ / CUDA 代码阅读问答](../doc/day09/day09_cpp_cuda_notes.md)
- [Release benchmark 原始 CSV](../doc/day09/results/stream_pipeline_release.csv)
- [2-stream Debug 正确性输出](../doc/day09/results/2stream_correctness_debug.txt)
- [GPU activity 明细 CSV](../doc/day09/results/day09_cuda_gpu_trace.csv)
- [原始 Nsight Systems 报告](day09_timeline.nsys-rep)
- [截图目录](../doc/day09/images/)

## Release benchmark 核心结果

固定 `N=16,777,219`、`block=256`，20 轮预热、50 轮正式测量，报告端到端中位数：

| chunk size | 1 stream | 2 streams | 4 streams | 本组最快 |
|---:|---:|---:|---:|---:|
| 65,536 | 69.615000 ms | 34.901750 ms | 22.573100 ms | 4 streams，3.084x |
| 262,144 | 21.297850 ms | 16.429450 ms | 14.338550 ms | 4 streams，1.485x |
| 1,048,576 | 14.759050 ms | 12.832200 ms | 12.023850 ms | 4 streams，1.227x |

9 组全部 `PASS`，最大绝对误差和最大相对误差均为 0。

## Nsight Systems 固定配置

- Release；
- `N=16,777,219`，`block=256`；
- 4 streams；
- `chunk_size=1,048,576`，共 17 chunks；
- capture range 外预热 1 轮，范围内采集 1 轮；
- 每条 stream 尾部记录 completion event，Host 等待全部 4 个 Event。

## GPU 活动摘要

| 活动 | 占比 | 总时间 | 实例数 | 中位数 |
|---|---:|---:|---:|---:|
| H2D | 50.0% | 5.827001 ms | 17 | 356.956 μs |
| D2H | 47.6% | 5.550110 ms | 17 | 339.163 μs |
| `simpleKernel` | 2.4% | 0.280763 ms | 17 | 17.695 μs |

传输占 GPU 活动时间 97.6%，累计传输时间约为累计 kernel 时间的 40.5 倍。

## 时间线结论

| 重叠类型 | 结果 |
|---|---:|
| H2D 与 kernel | 0 |
| kernel 与 D2H | 0 |
| H2D 与 D2H | 0 |
| 不同 stream 的 kernel 与 kernel | 12 对，最大 3.040 μs |

本配置没有 copy-compute overlap，但存在少量 concurrent kernel。Release benchmark 的加速不能归因于 copy-compute overlap。

## 同步审计

`cuda_api_sum` 中出现 4 次 `cudaEventSynchronize` 和 4 次 `cudaEventRecord`，分别对应 4 条活动 stream 的尾部完成边界。没有 `cudaDeviceSynchronize` 或 `cudaStreamSynchronize`。

`cudaMemcpyAsync` 的 CPU API duration 只表示提交开销；GPU 拷贝实际耗时应读取 CUDA HW 时间线或 `cuda_gpu_sum`。

## Day 9 验收结论

- 串行 pageable、串行 pinned、单 stream 异步和多 stream 分块路径正确；
- chunk、offset、尾块字节数与尾块 grid 均已验证；
- 1/2/4-stream benchmark 和原始数据已保存；
- pinned buffer、stream、event 与统计工具已 RAII / 公共化；
- Nsight Systems 报告、CSV、截图和分析结论齐全；
- 下一节点：Day 10 核心 Kernel 工具箱 v0.1 收束。
