# Day 4 补充：Benchmark 代码与指标推演

## 1. Benchmark 的四个阶段

`benchmarks/bench_vector_add.cu` 可以按职责拆成四段：

```text
准备：分配 Host/Device 数据，生成固定输入和 reference
验证：运行一次 GPU 计算并比较结果
测量：分别采集 kernel-only 与 end-to-end 样本
记录：计算中位数、有效带宽并输出 CSV
```

把职责分开能防止结果比较、打印或随机数生成进入 kernel 时间。

## 2. Event 计时逐行含义

概念代码：

```cpp
CudaEventTimer timer;
timer.start();
vector_add_device(d_a, d_b, d_c, n, block_size);
timer.stop();
samples.push_back(timer.elapsed_ms());
```

执行顺序：

1. start Event 被放入默认 stream；
2. kernel 被放到同一 stream，排在 start 之后；
3. stop Event 排在 kernel 之后；
4. `elapsed_ms()` 等待 stop 完成；
5. CUDA 计算两个 Event 时间戳之差。

`vector_add_device` 中的 `cudaGetLastError()` 检查启动状态，不会像 `cudaDeviceSynchronize()` 一样把每次 kernel 强制扩展为全设备同步。

## 3. Host 端到端计时逐行含义

概念代码：

```cpp
const auto begin = std::chrono::steady_clock::now();
cudaMemcpy(... H2D A ...);
cudaMemcpy(... H2D B ...);
vector_add_device(...);
cudaMemcpy(... D2H C ...);
const auto end = std::chrono::steady_clock::now();
```

使用 `steady_clock` 是因为它保证时间单调前进，系统时间校准不会让间隔突然倒退。

D2H 的阻塞式拷贝让 CPU 等待结果，因此 end 能代表这条同步数据路径结束。

## 4. 中位数实现

一般步骤：

```cpp
std::sort(samples.begin(), samples.end());
```

- 奇数个样本：取正中间；
- 偶数个样本：取中间两个的平均值。

项目使用 50 个样本，因此会平均排序后第 25 和第 26 个值（以人类从 1 开始计数）。

## 5. 有效带宽手算

假设：

```text
N = 1,000,000
kernel_ms = 0.025
```

则：

```text
bytes = 3 × 1,000,000 × 4 = 12,000,000 bytes
seconds = 0.025 / 1000 = 0.000025 s
GB/s = 12,000,000 / 0.000025 / 1e9
     = 480 GB/s
```

如果误把毫秒直接当秒，结果会小 1000 倍。这是常见单位错误。

## 6. 阅读 CSV 的顺序

1. 先检查 `correct` 是否全部 PASS；
2. 按相同 `size` 比较不同 `block`；
3. 小规模先观察固定开销，不急着下带宽结论；
4. 大规模观察 `kernel_ms_median` 与 `GB_per_s` 是否进入稳定区间；
5. 对比 `e2e_ms_median`，判断传输是否主导完整路径；
6. 只对当前设备、当前构建和当前样本范围下结论。

## 7. 复现实验检查单

- [ ] 使用 Release 构建；
- [ ] 关闭会显著占用 GPU 的其他程序；
- [ ] 保持 sizes、block sizes、seed、warmup、iters 不变；
- [ ] 保存原始 CSV，不只保存截图；
- [ ] 记录 GPU、CUDA 和日期；
- [ ] 性能异常时先重新验证 correct；
- [ ] 若要声称稳定提升，增加独立进程重复运行。
