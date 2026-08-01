# Day 4｜正确性测试与可信 Benchmark

> 补档状态：**当前 benchmark 骨架和单次补采数据完整，历史稳定性证据不完整**。2026-08-01 使用 Release 配置重新采集了 5 个规模、3 个 block size、20 次预热和 50 次测量的数据；尚未补做“三次独立进程运行”的波动对照。

## 1. 学习目标

- 把正确性验证与性能计时分开；
- 理解 kernel-only 与端到端时间回答不同问题；
- 使用 CUDA Event 测量 GPU stream 内时间；
- 使用 Host 稳定时钟测量 H2D + kernel + D2H；
- 通过预热、重复采样和中位数降低首次运行与偶然波动影响；
- 输出可机器读取的 CSV 和完整误差信息。

## 2. 公共组件

### 固定随机输入

`init_random()` 使用显式 seed，使多次运行生成相同输入：

```text
A seed = 20260729
B seed = 20260730
```

### 误差比较

`compare_results(actual, expected, atol, rtol)` 使用：

```text
abs_error = |actual - expected|
tolerance = atol + rtol × |expected|
```

同时记录：

- 是否通过；
- 首个错误位置；
- 最大绝对误差；
- 最大相对误差；
- 非有限值和长度不匹配。

### CUDA Event 计时

`CudaEventTimer` 将 start/stop Event 记录在同一 stream，并等待 stop Event 后计算 elapsed time。它测量的是 GPU 时间线中两个 Event 之间的工作，不包含 Host 端数据准备和普通 CPU 代码。

### Host 端到端计时

`std::chrono::steady_clock` 包围：

```text
H2D(A) + H2D(B) + vector add kernel + D2H(C)
```

最后的阻塞式 D2H 形成同步点，因此 stop 时间不会早于 GPU 结果完成。

## 3. Benchmark 配置

| 配置 | 值 |
|---|---|
| 构建 | Release |
| GPU | RTX 2080 Ti，Compute Capability 7.5 |
| CUDA | 13.3 |
| sizes | 1024、65536、1000003、4194304、16777217 |
| block | 64、128、256 |
| warmup | 20 |
| measured iterations | 50 |
| 汇总 | 中位数 |
| correctness | CPU reference，`atol=1e-6`、`rtol=1e-5` |

Vector add 每个元素读取 A、读取 B、写入 C，有效数据量定义为：

```text
transferred_bytes = 3 × N × sizeof(float)
GB/s = transferred_bytes / kernel_seconds / 1e9
```

这是程序根据算法有效数据量计算的**有效带宽**，不是 profiler 直接观测的 DRAM 实际字节数。

## 4. 代码文件

| 文件 | 作用 |
|---|---|
| `benchmarks/bench_vector_add.cu` | 预热、Event/Host 计时、中位数、CSV 和正确性 |
| `include/common/cuda_timer.h` | CUDA Event RAII 计时器 |
| `include/common/compare.h` | 绝对/相对误差与首个 mismatch |
| `include/common/init_data.h` | 固定 seed 随机输入 |
| `kernels/vector_add.cu` | 被测 grid-stride kernel |

## 5. 构建与运行

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

.\build\windows-ninja\Release\bench_vector_add.exe
```

原始补采数据：[day04_vector_add_baseline.csv](day04_vector_add_baseline.csv)

## 6. 核心结果

### 最大规模 `N=16,777,217`

| block | kernel 中位数 | 端到端中位数 | 有效带宽 | 正确性 |
|---:|---:|---:|---:|---|
| 64 | 0.379376 ms | 39.312698 ms | 530.68 GB/s | PASS |
| 128 | 0.374288 ms | 39.871552 ms | 537.89 GB/s | PASS |
| 256 | 0.373872 ms | 39.088303 ms | 538.49 GB/s | PASS |

所有 15 组配置均为 PASS，vector add 与 CPU 执行相同的单次加法，本次输出的最大绝对/相对误差均为 0。

### 如何解释

- `N=1024` 时 kernel 只有约 0.0044 ms，固定 launch/Event 开销占比很高，有效带宽只有约 2.8 GB/s，不能用它判断显存带宽上限；
- 随规模增大，有效带宽逐步上升，在最大规模达到约 530–538 GB/s；
- 最大规模下 block 64/128/256 的差距约 1.5%，说明三者都能提供足够并行度；当前数据不支持“某个 block 在所有规模上绝对最好”；
- 端到端时间约 39 ms，远大于约 0.374 ms 的 kernel-only 时间，因为端到端还包含 pageable Host/Device 传输；二者不能互相替代。

## 7. 指标含义与好坏判断

| 指标 | 含义 | 判断方式 |
|---|---|---|
| `kernel_ms_median` | 仅 GPU kernel 的 Event 中位时间 | 同工作量下越低越好 |
| `e2e_ms_median` | H2D + kernel + D2H 的 Host 中位时间 | 代表应用实际数据路径；同范围下越低越好 |
| `GB_per_s` | 基于 3×N×4 bytes 的有效带宽 | 只在相同算法数据量下比较，通常越高越好 |
| `max_abs_error` | 最大绝对偏差 | 必须与 atol/rtol 一起判断 |
| `max_rel_error` | 相对 expected 的最大偏差 | expected 很小时需由绝对阈值兜底 |

## 8. 已验证结论与局限

### 已验证

- 正确性与计时循环已经分离；
- 计时前有 20 次预热，报告 50 次中位数；
- kernel-only 与端到端时间的边界明确；
- CSV 包含 size、block、warmup、iters、时间、带宽和误差信息；
- 当前 15 组结果全部正确。

### 待补证据

- 当前只补采了一次完整进程输出，未保存三次独立运行的波动范围；
- CSV 没有单独列出 GPU/Driver/CUDA 字段，环境记录位于本文；
- 有效带宽不是硬件计数器，不能用于声称真实 DRAM 事务恰好等于 3×N×4 bytes。

## 9. 下一步

Day 5 保持同一计时框架，比较连续、跨步和 `float4` AXPY，第一次用数据解释全局内存访问模式。
