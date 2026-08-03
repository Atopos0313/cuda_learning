# Day 4：从“跑得快”到可信 Benchmark

> 本日目标：建立一套不会轻易误导自己的 CUDA 测量方法。先证明计算正确，再明确计时边界，理解 CUDA Event、Host 计时、预热、重复采样、中位数和有效带宽。

## 1. Benchmark 是什么

Benchmark 是一组**条件明确、可以重复、可以比较**的性能实验。它不只是调用一次计时函数。

一个可信的 CUDA benchmark 至少要回答：

1. 计算结果正确吗？
2. 测量的是 kernel，还是包含数据传输的完整流程？
3. 使用什么构建配置、输入规模和 block size？
4. 是否做过预热？
5. 重复多少次，用什么统计量汇总？
6. 两个版本是否完成同样工作、使用同样计时边界？

如果这些条件不清楚，“快了 20%”没有可验证含义。

## 2. 为什么必须先验证正确性

错误程序可能看起来特别快：

- 少处理了尾部元素；
- kernel 根本没有成功启动；
- 计时在 GPU 完成前结束；
- 输出没有拷回，也没有被检查；
- 优化版本改变了数学问题。

因此项目采用顺序：

```text
生成固定输入
  → CPU 计算 reference
  → GPU 计算
  → compare_results 验证
  → 只有 PASS 才解释性能
```

性能数字不能替代正确性证据。

## 3. CPU Reference 是什么

Reference 是一个以清晰、可信为首要目标的参考实现。向量加法的 CPU reference 为：

```cpp
for (std::size_t index = 0; index < n; ++index) {
    expected[index] = a[index] + b[index];
}
```

它不一定最快，但数学含义直接，适合判断 GPU 输出是否正确。

项目使用固定随机种子：

```text
A seed = 20260729
B seed = 20260730
```

固定 seed 的作用是让相同配置重复运行时得到相同输入，减少“输入变了”带来的干扰。固定 seed 不代表只测试一种规模；边界覆盖仍由 sizes 决定。

## 4. 浮点数为什么不能只用 `==`

许多浮点运算会因舍入顺序不同产生微小差异。项目用绝对误差和相对误差共同判断：

```text
absolute_error = |actual - expected|
tolerance = atol + rtol × |expected|

通过条件：absolute_error <= tolerance
```

### 4.1 绝对容差 `atol`

当 expected 接近 0 时，相对误差会被很小的分母放大，绝对容差提供一个固定的允许范围。

### 4.2 相对容差 `rtol`

当 expected 数值较大时，允许误差随结果尺度适度增长。

### 4.3 比较结果还记录什么

`CompareResult` 记录：

- `passed`：整体是否通过；
- `first_mismatch`：第一个超过容差的位置；
- `max_abs_error`：最大绝对误差；
- `max_rel_error`：最大相对误差；
- 长度不一致或非有限值也会判失败。

向量加法的 CPU/GPU 都执行一次 `float` 加法，本次实测误差为 0；但比较器仍使用一般化容差，以便后续 reduction 等运算顺序不同的算法复用。

## 5. CUDA 的异步性会怎样影响计时

Host 启动 kernel 时，通常只是把任务提交给 GPU：

```text
CPU：发出 kernel 启动命令 → 很快返回并继续
GPU：稍后开始执行 → 执行完成
```

错误的 Host 计时：

```cpp
auto begin = clock::now();
kernel<<<...>>>();
auto end = clock::now();
```

它可能主要测到 CPU 提交命令的耗时，而不是 GPU 执行 kernel 的耗时。必须通过 Event 或明确同步建立完成边界。

## 6. CUDA Event 是什么

CUDA Event 是插入某个 stream 时间线中的标记。可以把它理解成 GPU 工作队列里的路标：

```text
同一 stream：start event → kernel → stop event
```

当 stop event 完成时，排在它前面的 kernel 也已经完成。两 Event 的 GPU 时间戳差值就是这段 stream 工作的耗时。

项目的 `CudaEventTimer` 执行：

```cpp
timer.start(stream);
vector_add_device(..., stream);
timer.stop(stream);
const float milliseconds = timer.elapsed_ms();
```

`elapsed_ms()` 会先等待 stop Event，然后调用 `cudaEventElapsedTime`。

### 6.1 为什么 Event 必须记录在同一 stream

一个 stream 内的工作按顺序执行，因此 start、kernel、stop 的先后关系明确。如果随意把标记和 kernel 放到不同 stream，就不能直接用它们的排列证明只包围了目标工作。

Day9 会深入学习 stream；Day4 只需要记住：**计时 Event 和被测 kernel 使用同一个 stream**。

## 7. Kernel-only 时间测量了什么

本项目的 kernel-only 范围为：

```text
start Event
  → vector_add kernel
stop Event
```

它不包含：

- Host 随机数据生成；
- CPU reference；
- DeviceBuffer 首次申请；
- H2D 输入拷贝；
- D2H 输出拷贝；
- 结果比较和 CSV 输出。

这个指标适合比较 kernel 实现和 launch 配置。

## 8. End-to-End 时间测量了什么

本项目的端到端范围为：

```text
Host steady_clock start
  → H2D(A)
  → H2D(B)
  → kernel
  → D2H(C)
Host steady_clock stop
```

阻塞式 D2H 需要把结果交给 CPU，形成完成边界，因此 stop 不会早于相关 GPU 工作完成。

这个指标回答的是“输入已经在 Host、结果最终也要回 Host时，这条路径总共花多久”。它与 kernel-only 回答不同问题，不能相互替代。

## 9. 为什么计时前要预热

第一次运行可能包含一次性或冷启动开销，例如：

- CUDA 上下文初始化；
- 首次模块加载；
- 缓存和内存页面处于冷状态；
- GPU 时钟状态尚未稳定。

如果只测第一次，结果可能代表初始化而不是稳定执行。项目在正式采样前运行 20 次预热：

```text
20 次 warmup（不计入结果）
50 次 measured iterations（保存样本）
```

预热不是“作弊”，前提是明确写进实验条件。它测量的是稳态性能；如果研究首次请求延迟，就应该另设 cold-start 实验。

## 10. 为什么不能只测一次

单次测量会受操作系统调度、GPU 温度/频率、后台任务等偶然因素影响。重复采样能看到稳定趋势。

项目保存 50 个样本，排序后取中位数。

### 10.1 中位数是什么

把样本从小到大排序：

```text
0.37, 0.37, 0.38, 0.39, 1.20
```

中间的 `0.38` 是中位数。最后一个偶发的 `1.20` 不会像平均数那样明显拉高结果。

中位数并不是自动消除所有问题。严谨报告还可以记录最小值、最大值、百分位数和多次独立进程运行；当前 Day4 保存的是单次进程内 50 次采样的中位数。

## 11. 有效带宽怎样计算

向量加法每个元素：

- 从 A 读一个 `float`：4 bytes；
- 从 B 读一个 `float`：4 bytes；
- 向 C 写一个 `float`：4 bytes。

按算法有效数据量计算：

```text
transferred_bytes = 3 × N × sizeof(float)
kernel_seconds = kernel_ms / 1000
GB/s = transferred_bytes / kernel_seconds / 1e9
```

例：如果 `N = 1,000,000`，有效数据量约为 12 MB。

### 11.1 为什么叫“有效”带宽

这个值来自算法定义的数据量除以时间，不是 profiler 直接读取的 DRAM 事务字节数。缓存命中、写事务粒度等硬件行为可能让实际 DRAM 流量不同。

它适合在**相同工作量定义**下比较实现，不应被描述成硬件总线的精确实测流量。

## 12. 公平比较的控制变量

比较 block size 或 kernel 版本时，应保持：

- 同一 GPU 和软件环境；
- 同一 Release 构建；
- 同一输入数据与规模；
- 同一数学操作；
- 同一 warmup/iterations；
- 同一计时边界；
- 同一正确性阈值。

一次只改变一个主要因素，才更容易把性能变化归因给它。

## 13. 本日相关文件

| 文件 | 学习作用 |
|---|---|
| `benchmarks/bench_vector_add.cu` | 正确性、预热、Event/Host 计时、中位数和 CSV |
| `include/common/cuda_timer.h` | CUDA Event 的 RAII 计时封装 |
| `include/common/compare.h` | 绝对/相对误差与首个 mismatch |
| `include/common/init_data.h` | 固定 seed 输入 |
| `kernels/vector_add.cu` | 被测 grid-stride kernel |

逐项阅读 benchmark 代码见 [Benchmark 代码与指标推演](day04_benchmark_walkthrough.md)。

## 14. 实验配置

| 配置 | 值 |
|---|---|
| 构建 | Release |
| GPU | RTX 2080 Ti，Compute Capability 7.5 |
| CUDA | 13.3 |
| sizes | 1024、65536、1000003、4194304、16777217 |
| block sizes | 64、128、256 |
| warmup | 20 |
| measured iterations | 50 |
| 汇总 | 中位数 |
| correctness | CPU reference，`atol=1e-6`，`rtol=1e-5` |

## 15. 构建与运行

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

.\build\windows-ninja\Release\bench_vector_add.exe
```

原始补采数据：[day04_vector_add_baseline.csv](day04_vector_add_baseline.csv)

## 16. 核心结果

最大规模 `N = 16,777,217`：

| block | kernel 中位数 | 端到端中位数 | 有效带宽 | 正确性 |
|---:|---:|---:|---:|---|
| 64 | 0.379376 ms | 39.312698 ms | 530.68 GB/s | PASS |
| 128 | 0.374288 ms | 39.871552 ms | 537.89 GB/s | PASS |
| 256 | 0.373872 ms | 39.088303 ms | 538.49 GB/s | PASS |

全部 15 组配置均为 PASS，最大绝对误差和最大相对误差均为 0。

## 17. 怎样解释这些结果

### 17.1 小规模不能代表带宽上限

`N=1024` 时 kernel 约 0.0044 ms，launch/Event 等固定开销占比很高，有效带宽约 2.8 GB/s。此时主要问题是任务太小，不是显存本身只能提供这个带宽。

### 17.2 规模增大后固定开销被摊薄

数据规模增大时，有效带宽逐步升高；最大规模达到约 530～538 GB/s，说明足够多的线程和数据让设备进入更稳定的吞吐阶段。

### 17.3 三种 block size 差异很小

最大规模下最好与最差差距约 1.5%。当前证据说明 64、128、256 都能提供足够并行度；不能声称某个 block size 在所有规模和所有 GPU 上永远最佳。

### 17.4 端到端远大于 kernel-only

最大规模的 kernel 约 0.374 ms，而端到端约 39 ms。端到端还包括 pageable Host/Device 数据传输，因此二者差两个数量级并不矛盾。

## 18. 指标字典

| 字段 | 含义 | 使用方式 |
|---|---|---|
| `size` | 元素数量 N | 比较时确认工作规模一致 |
| `block` | 每 block 线程数 | 当前实验的主要变量 |
| `warmup` | 不计入结果的预热次数 | 描述稳态条件 |
| `iters` | 正式采样次数 | 样本数量 |
| `kernel_ms_median` | Event 测得的 kernel 中位时间 | 同工作量下通常越低越好 |
| `e2e_ms_median` | H2D+kernel+D2H 的 Host 中位时间 | 表示完整数据路径 |
| `GB_per_s` | 依据 `3*N*4` 计算的有效带宽 | 只在相同定义下比较 |
| `max_abs_error` | 最大绝对误差 | 与容差一起判断 |
| `max_rel_error` | 最大相对误差 | expected 很小时需结合绝对误差 |
| `correct` | reference 验证是否通过 | FAIL 时不解释性能 |

## 19. 常见误解

### 误解 1：kernel 启动语句前后用 CPU 时钟就是 kernel 时间

kernel 异步启动，未同步时通常只测到提交开销。

### 误解 2：最小的一次就是设备真实速度

最小值可能是偶然值。项目使用中位数代表本轮典型表现。

### 误解 3：Event 时间和端到端时间应当接近

它们包含的工作不同。只有先写清计时边界，数字才可解释。

### 误解 4：有效带宽等于 profiler 的 DRAM 实际流量

有效带宽是算法字节数模型，不是硬件事务计数器。

### 误解 5：Debug 结果也能代表优化后性能

Debug CUDA 构建可能启用设备调试信息和不同优化。性能对比应使用 Release，并记录配置。

## 20. 证据边界与完成标准

当前已验证：

- 正确性和计时循环分离；
- 20 次预热、50 次采样并报告中位数；
- kernel-only 与端到端计时边界明确；
- 15 组配置全部正确并保存 CSV。

尚未完成：

- 三次以上独立进程运行的波动范围；
- CSV 内独立记录 GPU、Driver、CUDA 字段；
- 用硬件计数器核对真实 DRAM 事务。

只有能解释上述测量边界、统计方法和结论限制，并能复现 CSV，才算完成 Day4。

下一步 Day5 将保持同一 benchmark 框架，只改变内存访问映射，观察连续访问、刻意跨步访问和 `float4` 向量访问为什么可能产生不同吞吐。
