# Day 5｜合并访存：第一个有证据的优化

> 补档状态：**正确性与 Release benchmark 已在 2026-08-01 补采；float4 在大规模下无收益的 profiler 根因仍待验证**。当前数据足以证明连续访问明显优于刻意跨步访问，但不把尚未采集的硬件原因写成定论。

## 1. 学习目标

- 观察同一 AXPY 数学操作在不同地址映射下的性能差异；
- 理解 warp 连续访问、跨步访问和内存事务浪费；
- 验证 `float4` 的 16-byte 对齐和不足 4 个元素的尾部；
- 用相同输入、相同工作量和相同 Event 计时范围比较版本。

AXPY 数学定义：

```text
y[i] = alpha × x[i] + y[i]
```

三个版本改变的是访问和指令组织，不改变数学结果。

## 2. 三种访问模式

### Contiguous

相邻线程处理相邻元素：

```text
lane 0 → x[0]
lane 1 → x[1]
...
lane 31 → x[31]
```

一个 warp 的地址连续，硬件更容易用较少内存事务服务请求。

### Strided

当前实现把线性空间解释为 `rows × 32`，再让相邻 lane 访问相隔 `rows` 个 float 的地址。数学上仍覆盖所有元素，但 warp 内地址分散，通常需要更多内存事务，降低有效带宽。

### Float4

`float4` 每次 load/store 处理连续 4 个 float，即 16 bytes。当前实现：

- 检查 Device 指针满足 `alignof(float4)`；
- 主体使用 grid-stride loop 处理 `n/4` 个向量；
- 仅由线程 0 处理 `n % 4` 的 0–3 个尾部元素；
- 对 `N=1、3、5、1,000,003` 等非 4 整数倍输入保留正确性。

向量化可能减少指令数量，但不保证更快；当连续标量版本已经接近带宽上限时，宽 load/store 可能没有额外收益。

## 3. 代码与测试

| 文件 | 作用 |
|---|---|
| `kernels/memory_access.cu` | contiguous、strided、float4 三个 AXPY kernel |
| `include/memory_access.h` | 三种 Device 启动接口 |
| `tests/test_memory_access.cu` | 3 variants × 6 sizes × 3 block sizes 的正确性 |
| `benchmarks/bench_memory_access.cu` | 固定输入、单次 reference 检查、预热、Event 中位数和 CSV |

正确性尺寸：

```text
N = 0、1、3、4、5、1,000,003
block = 64、128、256
3 个版本，共 54 组
```

Release CTest 中 `memory_access_correctness` 为 PASS。

## 4. Benchmark 配置

| 配置 | 值 |
|---|---|
| GPU | RTX 2080 Ti，CC 7.5 |
| CUDA | 13.3 |
| sizes | 1,000,003、4,194,304、16,777,217 |
| block | 64、128、256 |
| warmup | 20 |
| iterations | 50 |
| 汇总 | kernel Event 中位数 |
| alpha | 1.1 |
| correctness | `atol=1e-6`、`rtol=1e-5` |

有效带宽仍按每个元素读 x、读 y、写 y 计算：

```text
GB/s = 3 × N × sizeof(float) / kernel_seconds / 1e9
```

## 5. 构建与运行

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

.\build\windows-ninja\Release\test_memory_access.exe
.\build\windows-ninja\Release\bench_memory_access.exe
```

原始补采数据：[day05_memory_access.csv](day05_memory_access.csv)

## 6. 核心结果

下表选择每个版本在各规模中的最佳实测 block 配置；跨版本因果比较还会结合相同 block 的数据。

| N | 版本 | 最佳 block | kernel 中位数 | 有效带宽 | 正确性 |
|---:|---|---:|---:|---:|---|
| 1,000,003 | contiguous | 128 | 0.026144 ms | 459.00 GB/s | PASS |
| 1,000,003 | strided | 256 | 0.069632 ms | 172.34 GB/s | PASS |
| 1,000,003 | float4 | 64 | 0.020800 ms | 576.92 GB/s | PASS |
| 4,194,304 | contiguous | 128 | 0.095024 ms | 529.67 GB/s | PASS |
| 4,194,304 | strided | 256 | 0.306288 ms | 164.33 GB/s | PASS |
| 4,194,304 | float4 | 256 | 0.095472 ms | 527.19 GB/s | PASS |
| 16,777,217 | contiguous | 64 | 0.366144 ms | 549.86 GB/s | PASS |
| 16,777,217 | strided | 256 | 1.641056 ms | 122.68 GB/s | PASS |
| 16,777,217 | float4 | 256 | 0.370800 ms | 542.95 GB/s | PASS |

### 固定 block=256 的公平对比

| N | contiguous | strided | float4 | 观察 |
|---:|---:|---:|---:|---|
| 1,000,003 | 0.026240 ms | 0.069632 ms | 0.021488 ms | strided 慢约 2.65×；float4 快约 1.22× |
| 4,194,304 | 0.095056 ms | 0.306288 ms | 0.095472 ms | strided 慢约 3.22×；float4 基本持平 |
| 16,777,217 | 0.366816 ms | 1.641056 ms | 0.370800 ms | strided 慢约 4.47×；float4 约慢 1.1% |

## 7. 证据与解释

### 已验证观察

- 27 个 benchmark 配置全部 PASS；最大绝对误差 `2.38e-07`、最大相对误差 `1.19e-07`，低于设定阈值；
- 大规模下 contiguous 明显快于 strided；
- strided 的 block 256 优于 64，但即使取其最佳配置仍远慢于 contiguous；
- float4 在约 100 万元素时更快，但在 419 万和 1677 万元素时与 contiguous 持平或略慢；
- 非 4 整数倍长度通过 correctness，尾部逻辑有效。

### 机制解释

连续访问让 warp lane 请求相邻地址，更容易形成高效的内存事务；刻意跨步映射让地址分散，同样的有效数据量需要更多事务和等待，因此有效带宽显著下降。

### 初步判断，尚待 profiler 验证

大规模 contiguous 已达到约 550 GB/s，float4 没有继续提高有效带宽，初步判断可能是标量版本已经接近当前 kernel/设备的内存吞吐上限，减少部分指令不足以突破带宽限制。没有 Day 5 的 NCU 计数器或时间线证据，因此这里只标记为假设。

## 8. 指标边界与常见误区

- 高 `GB/s` 只有在数学工作量和有效字节定义一致时才可比较；
- strided 的低有效带宽不等于 DRAM 没有传输数据，反而可能包含更多无效事务；
- `float4` 需要正确对齐和尾部处理；直接 reinterpret_cast 不等于安全；
- 向量化不是性能保证，最终要回到相同 workload 的普通 benchmark；
- 当前测试没有保存 Day 5 profiler 报告，不能把“带宽饱和”升级为已验证根因。

## 9. 已完成与待补

### 已完成

- 三种访问模式实现和 correctness；
- 非 4 整数倍尾部验证；
- 3 个规模 × 3 个 block × 3 个版本的 Release CSV；
- 连续/跨步性能差异和 float4 收益边界分析。

### 待补

- 使用 Nsight Compute 采集内存事务、吞吐与 stall 证据；
- 解释 float4 在大规模下无收益的硬件根因；
- 将 profiler 结果与当前普通 benchmark 组成完整证据链。

## 10. 下一步

Day 6 进入矩阵转置，继续观察全局内存合并访问，并引入 shared memory tiled transpose 与 bank conflict。
