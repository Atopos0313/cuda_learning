# Day 07：CUDA Reduction、全 GPU 多轮归约与 Warp Shuffle

## 1. 当天主题与完成状态

Day 07 从“一个 block 如何得到一个部分结果”出发，逐步完成了完整的 CUDA
reduction 学习闭环：

1. shared memory block 内树形 sum；
2. 尾部不足一个 block 时使用单位元补齐；
3. GPU 输出多个部分和，CPU 使用 double 合并；
4. 使用 ping-pong buffer 在 GPU 上继续归约，直到只剩一个结果；
5. 将合并操作从加法替换为 `fmaxf`，实现 max reduction；
6. 使用 `__shfl_down_sync` 优化最后一个 warp；
7. 使用 CUDA Event 对 shared 与 warp 两个版本进行 Release benchmark；
8. 保存原始 CSV，并分析正确性、时间、吞吐量和 speedup。

Day 07 已完成。专题文档：

- [Part 1：Shared Memory 分块归约](day07_part1_shared_memory_reduction_analysis.md)
- [Part 2：全 GPU 多轮归约与 Ping-Pong Buffer](day07_part2_gpu_multistage_reduction_analysis.md)
- [Part 3：Max、Warp Shuffle 与 Benchmark](day07_part3_max_warp_shuffle_benchmark_analysis.md)
- [原始 benchmark CSV](day07_reduction.csv)

## 2. 实验目的

本日实验回答四个问题：

1. 为什么 reduction 不能像 elementwise kernel 一样让每个线程独立完成？
2. 为什么一个普通 kernel 只能先得到 block 部分结果？
3. 如何让中间结果留在 GPU，并通过多轮 kernel 得到最终结果？
4. 当只剩一个 warp 时，为什么 shuffle 能替代 shared memory 和多次 block 同步？

最终目标是同时得到 sum 和 max 的 shared 基线及 warp 优化版本，并用相同输入、
相同 block、相同归约轮数验证正确性与性能。

## 3. 核心原理

### 3.1 树形归约

以 8 个数为例，参与合并的线程数逐轮减半：

```text
8 个输入
→ stride=4：4 个部分结果
→ stride=2：2 个部分结果
→ stride=1：1 个 block 结果
```

sum 的合并操作是：

```cpp
a + b
```

max 的合并操作是：

```cpp
fmaxf(a, b)
```

### 3.2 单位元

尾部越界线程也必须向 shared memory 写入安全值：

```text
sum：0，因为 x + 0 = x
max：-∞，因为 max(x, -∞) = x
```

max 不能填 0，否则纯负数输入可能错误地得到 0。

### 3.3 为什么需要多轮 kernel

`__syncthreads()` 只能同步一个 block，不能同步整个 grid。第一轮只能得到：

```text
block 0 → partial[0]
block 1 → partial[1]
...
```

host 在同一个 stream 中继续启动相同 kernel：

```text
100003 → 391 → 2 → 1
```

同一 stream 中的 kernel 按提交顺序执行，因此上一轮完成后下一轮才会读取结果。

### 3.4 Ping-Pong Buffer

两块临时 GPU 缓冲区交替承担输入和输出：

```text
d_input → d_ping → d_pong → d_ping
```

不能让不同 block 在同一缓冲区中原地读写，因为 block 之间没有全局同步，某个
block 可能提前覆盖另一个 block 尚未读取的数据。

### 3.5 Warp Shuffle

shared 基线在最后一个 warp 中仍然执行 shared-memory 读写和
`__syncthreads()`。优化版在只剩 32 个值后执行：

```text
offset=16 → 8 → 4 → 2 → 1
```

`__shfl_down_sync` 让同一个 warp 的线程直接读取其他 lane 的寄存器值。结果保留
在 lane 0 的寄存器中，不需要写回 shared memory 后再读取。

## 4. 核心代码及职责

- [`include/reduction.h`](../../include/reduction.h)
  - 声明 shared 与 warp 版本的 sum/max launcher。
- [`kernels/reduction.cu`](../../kernels/reduction.cu)
  - shared-memory sum/max kernel；
  - warp 内 sum/max 辅助函数；
  - warp-shuffle sum/max kernel；
  - block 参数验证与 kernel launch。
- [`tests/test_reduction.cu`](../../tests/test_reduction.cu)
  - CPU reference；
  - 第一阶段 CPU 合并版本；
  - 全 GPU ping-pong 版本；
  - shared/warp 正确性对比；
  - 空输入、边界、不规则规模、纯负数与极端浮点测试。
- [`benchmarks/bench_reduction.cu`](../../benchmarks/bench_reduction.cu)
  - CUDA Event 计时；
  - shared/warp 的 sum/max pipeline 对比；
  - median、吞吐量、有效带宽、speedup 和误差输出。
- [`CMakeLists.txt`](../../CMakeLists.txt)
  - 将 reduction kernel、test 和 benchmark 加入构建。

## 5. 构建与运行

Release 构建：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build.ps1 `
  -Configuration Release
```

正确性测试：

```powershell
.\build\windows-ninja\Release\test_reduction.exe
```

benchmark：

```powershell
.\build\windows-ninja\Release\bench_reduction.exe
```

保存 CSV：

```powershell
.\build\windows-ninja\Release\bench_reduction.exe |
  Set-Content -Encoding utf8 .\doc\day07\day07_reduction.csv
```

本日按学习约定没有运行 Sanitizer 或 Synccheck。

## 6. Benchmark 环境与范围

| 项目 | 本次设置 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti |
| CUDA Toolkit | 13.3 |
| CUDA architecture | sm_75 |
| 构建 | Release |
| OS | Windows |
| Host compiler | MSVC 14.44.35207 |
| dtype | float |
| 预热 | 20 次 |
| 正式计时 | 50 次 |
| 统计量 | median |
| 输入规模 | 1003、100003、1048576、16777216 |
| block | 32、64、128、256 |

计时范围包括 GPU 上从 `N` 个输入到 1 个结果的全部 kernel，不包括：

- device allocation；
- H2D；
- 最终一个 float 的 D2H；
- CPU reference。

因此本数据是 GPU pipeline 时间，不是应用端到端时间。

## 7. CSV 指标解释

| 指标 | 含义 | 判断 |
|---|---|---|
| `operation` | `sum` 或 `max` | 分类字段 |
| `variant` | `shared` 基线或 `warp` 优化版 | 同条件比较 |
| `N` | 原始输入元素数 | workload 规模 |
| `block` | 每个 block 的线程数 | 不是越大越好 |
| `rounds` | 从 N 缩减到 1 的 kernel 数 | 通常越少越有利，但不是唯一因素 |
| `warmup` | 正式计时前的预热次数 | 消除冷启动影响 |
| `iters` | 正式计时次数 | 本次为 50 |
| `pipeline_ms_median` | 完整 GPU 归约 pipeline 中位时间，毫秒 | 越低越好 |
| `speedup_vs_shared` | 同 operation、N、block 下的 `shared_ms / variant_ms` | 大于 1 表示更快 |
| `M_elements_per_s` | 每秒处理的百万个原始输入元素 | 越高越好 |
| `effective_GB_per_s` | 按每轮全局读写量计算的有效带宽 | 越高越好，不等于硬件实测 DRAM 带宽 |
| `reference` | CPU reference | 正确性基准 |
| `result` | GPU 最终结果 | 与 reference 比较 |
| `abs_error` | `|result-reference|` | sum 看容差；普通 max 应为 0 |
| `correct` | 正确性状态 | 必须为 PASS |

## 8. 核心结果

CSV 共 64 行结果，全部 `PASS`。

### 8.1 最大规模 N=16777216

#### Sum

| block | shared ms | warp ms | speedup |
|---:|---:|---:|---:|
| 32 | 0.803584 | 0.606992 | 1.324× |
| 64 | 0.405504 | 0.302720 | 1.340× |
| 128 | 0.394400 | 0.252400 | **1.563×** |
| 256 | 0.435584 | 0.282816 | 1.540× |

#### Max

| block | shared ms | warp ms | speedup |
|---:|---:|---:|---:|
| 32 | 0.803008 | 0.608864 | 1.319× |
| 64 | 0.394080 | 0.303168 | 1.300× |
| 128 | 0.386352 | 0.251904 | **1.534×** |
| 256 | 0.429248 | 0.282624 | 1.519× |

本次最高 speedup 为：

```text
sum, N=16777216, block=128
0.394400 ms → 0.252400 ms
speedup = 1.563×
```

### 8.2 规模效应

- `N=1003`：warp speedup 为约 1.058–1.134×；
- `N=100003`：约 1.100–1.178×；
- `N=1048576`：约 1.325–1.454×；
- `N=16777216`：约 1.300–1.563×。

所有 warp 行的简单平均 speedup 为 1.264×。该平均值只描述本次 32 个配置，
不是按真实业务 workload 加权的总体加速比。

## 9. 结果分析

### 已验证

1. shared 和 warp 版本在所有测试配置中正确；
2. max 的 CPU/GPU 结果完全相等，`abs_error=0`；
3. sum 的误差均满足既定的绝对误差加相对误差容限；
4. warp 版本在本次全部配置中快于同 block 的 shared 基线；
5. 大规模输入下，减少最后一个 warp 的 shared-memory 访问和 block 同步带来
   更明显收益；
6. block=128 在两个最大规模下取得本次最优或接近最优结果。

### 初步解释，仍需 profiler 验证

- 小规模只有微小收益，可能是因为 kernel launch、事件分辨率和固定开销占比高；
- block=32 产生更多 block 和更多归约轮数，因此大规模时明显慢于 64/128；
- block=256 虽然轮数较少，但没有超过 block=128，可能与调度、活跃 block 数、
  shared memory、指令吞吐或延迟隐藏有关；
- warp shuffle 的收益应主要来自减少 shared-memory 指令和 barrier，但需要
  Nsight Compute 的指令及 barrier 指标才能确认具体贡献。

## 10. 遇到的问题与修复

### PowerShell 禁止执行脚本

直接运行 `build.ps1` 被 execution policy 拦截。使用一次性子进程参数：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ...
```

没有修改系统全局策略。

### `CUDART_INF_F` 未定义

max padding 需要负无穷单位元。最初只包含 `cuda_runtime.h` 时常量未暴露，显式
加入：

```cpp
#include <math_constants.h>
```

之后使用：

```cpp
-CUDART_INF_F 表示float类型的负无穷大
```

### Warp mask 与 block 约束

`0xffffffffu` 表示 32 个 lane 全部参与。因此 warp 版本要求 block 至少 32 个
线程，当前树形算法还要求 block 是 2 的幂。launcher 将合法范围限制为：

```text
32、64、128、256、512、1024
```

## 11. 最终结论、局限与下一步

Day 07 已验证：

- shared memory 能完成 block 内树形 reduction；
- kernel 边界能形成跨 block 的全局阶段分隔；
- ping-pong buffer 能让多轮中间结果留在 GPU；
- sum 与 max 可以共享 reduction 框架，只需替换操作和单位元；
- warp shuffle 在本机大规模 workload 上取得最高 1.563× pipeline speedup。

局限：

- 当前只支持 `float`；
- warp 版本要求 block 为 `[32,1024]` 范围内的 2 的幂；
- max 尚未定义包含 NaN、正负零时的业务语义；
- benchmark 不包含 H2D、D2H 和分配，不能代表端到端加速；
- 尚未使用 profiler 证明具体指令和 barrier 变化。

下一步进入 Day 08：使用 Nsight Compute 为 copy、transpose 和 reduction 建立
固定 profiling case，并完成一次“指标证据 → 瓶颈假设 → 代码修改 → 结果验证”。
