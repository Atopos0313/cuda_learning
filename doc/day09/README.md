# Day 9｜Stream、Pinned Memory 与拷贝计算重叠

> 2026-08-01 学习停止点：**Day 9 阶段未完成**。已完成小阶段 1“串行 pageable/pinned 正确性基线”和小阶段 2“单 stream 异步链”；多 stream chunking、性能对比与 Nsight Systems 时间线仍待学习和验证。

## 今日工程目标

构造 `H2D → kernel → D2H` 数据流水线，逐步从同步串行版本扩展到 pinned memory、异步拷贝和多个非默认 stream，最终使用 Nsight Systems 时间线判断拷贝与计算是否真的重叠。

## 小阶段 1｜串行 pageable/pinned 正确性基线（已完成）

### 完成条件

- 保留普通 pageable host memory 路径；
- 新增 pinned host memory 路径；
- 两条路径使用相同输入、kernel、设备缓冲区和误差阈值；
- 使用不能整除 block size 的元素数量验证尾部边界；
- 两条路径均通过 CPU reference 检查；
- pinned buffer 在 GPU 工作完成后释放。

### 概念定义

**Pageable memory** 是普通 `std::vector` 使用的可分页主机内存，操作系统可以管理和换出这些内存页。CUDA Runtime 在主机与 GPU 之间拷贝这类内存时，可能需要先经过内部 pinned 中转区。

**Pinned memory** 是由 `cudaMallocHost` 分配的锁页主机内存。它不会在使用期间被操作系统换出，GPU 的 DMA 拷贝引擎可以稳定访问它，是后续可靠使用异步传输与传输/计算重叠的必要条件之一。

Pinned memory 会占用有限的系统锁页资源，因此应当一次分配、重复使用，并在所有相关 GPU 工作完成后再释放；不应把频繁分配释放的成本混入传输 benchmark。

### 当前执行流

两条路径目前都使用同步拷贝和默认 stream：

```text
CPU 准备输入
→ 阻塞式 H2D
→ kernel 启动
→ 阻塞式 D2H 等待此前 kernel 完成
→ CPU 正确性检查
```

代码没有在 kernel 后额外调用 `cudaDeviceSynchronize()`。阻塞式 D2H 已经形成必要同步点；额外的全设备同步既不需要，也会干扰后续对不同同步方式的比较。

### 代码与配置

- `apps/stream_pipeline.cu`：pageable/pinned 串行路径、简单变换 kernel 与正确性检查。
- `CMakeLists.txt`：登记 `stream_pipeline` 可执行目标。
- `include/common/device_buffer.h`：管理 device memory。
- `include/common/compare.h`：比较 GPU 输出与 CPU reference。

固定配置：

```text
元素数量：1,000,003
threads per block：256
grid：ceil(N / 256)
kernel：output[i] = input[i] * 2 + 1
随机种子：20260801
绝对误差阈值：1e-6
相对误差阈值：1e-5
```

`1,000,003` 不能被 256 整除，因此最后一个 block 含有越界线程；kernel 中的 `index < n` 负责保护尾部。

### 构建与运行

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Debug
.\build\windows-ninja\Debug\stream_pipeline.exe
```

本机当前 PowerShell 执行策略会直接拦截 `.\scripts\build.ps1`；上面的 `Bypass` 只作用于本次子进程，不修改系统级执行策略。

### 实际输出

```text
mode=serial_pageable, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
mode=serial_pinned, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
```

建议后续将第一行模式名改为 `serial_pageable`，使正式结果不产生歧义。

### 已验证结论

- pageable 与 pinned 路径均正确处理了不完整 block 尾部；
- 两条路径与相同 CPU reference 完全一致；
- pinned 指针确实用于 H2D/D2H，而不是只完成了分配；
- 正常执行路径中，pinned buffer 在阻塞式 D2H 和结果读取完成后释放；
- 本阶段只验证正确性，未建立预热、重复计时和固定计时范围，因此不能声称 pinned 更快。

### 常见误区与局限

- `cudaMallocHost` 本身不会让同步程序自动发生重叠；它只是为真正的异步传输提供必要条件。
- kernel 启动通常对 CPU 异步，但后续阻塞式 D2H 会等待默认 stream 中排在它之前的 kernel。
- `cudaGetLastError()` 检查 kernel 启动错误，不等同于等待 kernel 完成。
- 当前手动调用 `cudaFreeHost` 的方式在异常路径上不具备完整 RAII 保障；正式 benchmark 前应改为自动生命周期管理。

## 小阶段 2｜Stream、Event 与单 stream 异步链（已完成）

### 完成条件

- 使用一个 `cudaStreamNonBlocking` 非默认 stream；
- pinned H2D、kernel 和 pinned D2H 全部提交到同一 stream；
- 在 D2H 后记录 completion event，并由 CPU 等待该 event；
- Event 完成前不读取或释放 pinned output；
- 不使用 `cudaDeviceSynchronize()`；
- pageable 串行、pinned 串行和单 stream 异步三条路径全部通过相同 reference 检查。

### Stream 是什么

CUDA Stream 是 CPU 向 GPU 提交任务的一条有序队列。同一 stream 内的操作按提交顺序执行：

```text
H2D → kernel → D2H
```

Stream 不是 CPU 线程、GPU 核心或固定硬件引擎。创建 stream 只提供组织任务和潜在并发的条件，不保证任务一定并行。单 stream 内的本实验仍然严格串行。

本实验使用 `cudaStreamNonBlocking` 创建非默认 stream，避免它与 legacy default stream 建立额外的隐式同步关系。

### Event 是什么

CUDA Event 是插入 stream 的进度标记：

```text
H2D → kernel → D2H → completion event
```

只有 Event 之前的任务全部完成，Event 才会进入完成状态。因此 `cudaEventSynchronize(completion_event)` 返回后，CPU 才能安全读取 D2H 输出。

本阶段使用 `cudaEventDisableTiming`，因为 completion event 只负责同步，不负责计时。与等待整个设备的 `cudaDeviceSynchronize()` 相比，Event 能表达更精确的 stream 内完成边界。

### 实际执行链

```text
CPU：提交 memset → 提交 H2D → 提交 kernel → 提交 D2H → 记录 Event → 等待 Event
GPU：       memset →      H2D →      kernel →      D2H → Event 完成
```

Host output 在提交异步任务前完整填为 `-1`，device output 在同一 stream 中清零。这样可以避免沿用上一轮正确结果造成假 PASS。

Kernel 使用四参数启动形式：

```text
<<<grid, block, dynamic_shared_memory_bytes, stream>>>
```

当前动态共享内存字节数为 0，第四个参数指定非默认 stream。

### 构建与验证

使用 Debug 配置重新生成并构建 `stream_pipeline`，实际运行输出：

```text
mode=serial_pageable, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
mode=serial_pinned, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
mode=async_pinned, streams=1, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
```

### 已验证结论

- 三条路径均正确处理 `N=1,000,003` 的不完整 block 尾部；
- H2D、kernel、D2H 和 completion event 位于同一非默认 stream，顺序正确；
- CPU 在 Event 完成后才构造 `async_result`；
- Event 和 stream 销毁后才释放 pinned buffer，没有过早释放；
- 本阶段证明了单 stream 异步链的正确性，但没有证明传输与计算重叠，也没有建立性能结论。

### 常见误区与局限

- `Async` 表示 API 可以在 GPU 工作完成前返回，不表示同一 stream 内的任务自动重叠。
- `cudaEventRecord` 只是把标记加入 stream；调用返回时 Event 不一定已经完成。
- `cudaGetLastError()` 不等待 kernel 完成。
- 多 stream 也只提供重叠机会；硬件 copy engine、依赖关系和 chunk 大小会决定实际时间线。
- 当前手动资源管理在 CUDA API 抛出异常的路径上仍可能泄漏；正式 benchmark 前需要 RAII 化。

## 扩展理解｜多 Stream 并发、资源限制与数据安全

### GPU 可以处理多个 Stream，但不保证同时执行

多个 stream 让 GPU 同时看到多条独立的有序任务队列：

```text
Stream 0：H2D(A) → Kernel(A) → D2H(A)
Stream 1：H2D(B) → Kernel(B) → D2H(B)
```

这只提供并发执行的机会。实际是否重叠还取决于：

- GPU 是否支持 copy/compute overlap、concurrent kernels 或并发传输；
- 不同任务之间是否存在数据依赖；
- SM 上是否还有线程、warp、block、寄存器和 shared memory 容量；
- copy engine 是否可用；
- Host 提交顺序以及是否插入了隐式或显式同步；
- kernel 和传输是否足够长，能够在时间线上观察到重叠。

因此不能从源码中出现两个 stream 就直接得出“已经并行”。最终证据必须来自 Nsight Systems 时间线。

### 为什么 Copy 与 Kernel 通常比两个 Kernel 更容易重叠

GPU 内部不是单一执行单元。简化后可以分为：

```text
Copy Engine：处理 H2D/D2H 数据传输
SM：执行 kernel 的 block、warp 和线程
```

当 `Kernel(chunk 0)` 使用 SM 时，`H2D(chunk 1)` 可能使用 copy engine，因此两者有机会重叠：

```text
Copy Engine：        H2D(chunk 1)
SM：          Kernel(chunk 0)
时间：        ───────────────→
```

两个 kernel 也可能并发，但前提是它们的 block 能同时驻留。GPU 调度的基本单位不是“把某个 kernel 固定分给若干个 SM”，而是逐步把 grid 中的 block 安排到具备足够资源的 SM 上。

如果 Kernel A 的驻留 block 已经占满所有可用线程、寄存器、shared memory 或 block 槽位，Kernel B 暂时不能进入；当某些 block 完成并释放资源后，B 才可能获得驻留机会。不能只看“有多少个 SM”，也不能把“使用两个 stream”理解成“两个 kernel 各分一半 SM”。

### Stream Priority 不是抢占开关

`cudaStreamCreateWithPriority` 可以给 stream 设置调度优先级，但应记住：

```text
高优先级 = 给等待中的工作更高调度倾向
高优先级 ≠ 暂停已经运行的低优先级工作
高优先级 ≠ 保证某个确定执行顺序
```

NVIDIA 当前文档明确将 stream priority 描述为调度提示：它主要影响等待中的 kernel 工作，不抢占已经运行的工作，也不为执行顺序提供功能保证。程序正确性不能依赖“高优先级 kernel 一定立刻插队”。

这里也不应扩展成“所有 GPU kernel 在所有平台上绝对不可抢占”。硬件、驱动和操作系统可能具备不同粒度的计算抢占能力；CUDA stream 编程层面真正可靠的规则是：**不要依赖抢占，使用明确同步和数据依赖保证正确性。**

### 数据竞争与 Kernel 是否被抢占是两个问题

假设两个 kernel 访问互不相关的数据：

```text
Kernel A：array_a[0] = 100
Kernel B：array_b[0] = 200
```

它们不存在共享数据冲突，但这不能推出 GPU 会暂停 A、立即运行 B，也不能保证两者并发。调度由 stream 依赖、优先级提示和硬件资源共同决定。

如果两个 kernel 无同步地修改同一位置：

```text
Kernel A：data[0] += 1
Kernel B：data[0] += 2
```

并发执行时可能出现以下问题：

- **Data Race（数据竞争）**：多个执行单元并发访问同一数据，至少一个是写入，而且缺少足以建立顺序的同步。
- **Race Condition（竞态条件）**：程序最终行为依赖不可控的执行时序；它是比 data race 更宽泛的问题。
- **Synchronization（同步）**：使用 stream 内顺序、Event、`cudaStreamWaitEvent`、barrier 或 atomic 等机制建立所需执行关系。

这两个问题必须分开：

| 问题 | 正确关注点 |
|---|---|
| Kernel B 何时获得执行机会？ | stream 依赖、优先级提示、block 调度和硬件资源 |
| A、B 并发访问同一数据是否正确？ | 数据依赖、data race 与同步机制 |

### Pageable、Pinned、DMA 与 Async 的完整关系

普通 `std::vector` 通常位于 pageable host memory。操作系统能够换出或重新映射这些内存页，CUDA 在传输时可能需要使用内部 pinned staging buffer：

```text
Pageable Host Memory
→ CPU/Runtime staging copy
→ 临时 Pinned Buffer
→ DMA
→ Device Memory
```

`cudaMallocHost` 分配 page-locked host memory，使 DMA 能稳定访问这块 Host 内存：

```text
Pinned Host Memory
→ DMA
→ Device Memory
```

因此可以这样记：

```text
Async API：提供异步提交能力
Pinned Memory：为可靠的 Host/Device 异步 DMA 提供条件
Stream：表达 GPU 任务顺序与潜在并发
Event：表达完成点和跨任务依赖
```

但不要把结论说成“函数名有 Async 就一定异步”或“pinned 后 API 永远立即返回”。CUDA Runtime 文档说明，pageable host memory 的异步传输可能相对 Host 同步并发生 staging；即使在通常应异步的情形下，CUDA API 也可能因内部资源竞争而阻塞，程序不能依赖未文档化的立即返回行为。

### 异步期间不能提前访问 Pinned Buffer

提交 pinned H2D 后，在传输完成前不能修改输入：

```text
cudaMemcpyAsync(H2D)
→ GPU 可能仍在读取 h_input_pinned
→ CPU 此时修改它会破坏传输数据
```

提交 pinned D2H 后，在传输完成前不能读取或释放输出：

```text
cudaMemcpyAsync(D2H)
→ GPU 可能仍在写 h_output_pinned
→ CPU 必须等待 completion event
→ 才能读取或释放
```

本阶段把 Event 放在完整的 `H2D → kernel → D2H` 之后，因此 Event 完成时，输入传输和输出传输都已经结束。以后为了更早复用输入 buffer，也可以在 H2D 后单独记录更细粒度的 Event。

### 本节最小记忆

```text
Stream 是有序任务队列，不是线程或 SM。
多 Stream 提供并发机会，不保证并发结果。
Priority 是调度提示，不会抢占已运行工作。
资源不足时，其他 kernel 的 block 必须等待可驻留资源。
数据竞争与调度/抢占是两个不同问题。
Async + Pinned + Stream 提供异步传输基础，是否重叠仍需时间线验证。
Event 完成前，不要读取、修改或释放相关 pinned buffer。
```

### 官方参考

- [CUDA Programming Guide：Asynchronous Execution、Streams、Priorities 与 Overlap](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA Runtime API：API synchronization behavior](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-runtime-api/api-sync-behavior.html)
- [CUDA Driver API：Stream priority 的调度边界](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html)
- [Nsight Systems Analysis Guide：pageable async memcpy 与同步诊断](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

## 下一小阶段｜Chunking 与多 Stream 调度模型

进入代码前先解释为什么需要把完整数组切成 chunk、每个 stream 为什么需要独立的 device buffer 或不重叠片段，以及最后一个不完整 chunk 如何计算偏移和长度。随后实现 2 个 stream 的最小正确性版本：

```text
Stream 0：H2D(chunk 0) → kernel(chunk 0) → D2H(chunk 0)
Stream 1：H2D(chunk 1) → kernel(chunk 1) → D2H(chunk 1)
```

先验证结果和尾部处理，再扩展到 1、2、4 个 stream 与多个 chunk size；是否发生重叠最终由 Nsight Systems 时间线判断。

## 2026-08-01 收尾记录

### 今日已完成

- 完成 pageable 串行、pinned 串行与单 stream 异步三条正确性路径；
- 掌握 stream 是有序任务队列、Event 是 stream 进度标记；
- 使用 completion event 保护异步 D2H 后的 Host 读取和 pinned buffer 生命周期；
- 区分多 stream 提供并发机会与硬件实际发生重叠；
- 梳理 stream priority、资源限制、data race、DMA 与 pageable/pinned staging 的边界。

### 今日未完成

- 尚未实现 chunking；
- 尚未实现 2 个或 4 个 stream；
- 尚未处理最后一个不完整 chunk；
- 尚未建立 1、2、4 stream 与多个 chunk size 的 Release benchmark；
- 尚未生成 `benchmarks/bench_stream_pipeline.cu`、`scripts/profile_nsys.ps1` 和 `reports/day09_timeline.nsys-rep`；
- 尚未使用 Nsight Systems 证明传输与计算是否重叠；
- 当前 pinned、stream 和 event 仍为手动资源管理，异常路径 RAII 化待完成。

### 下次继续入口

先从零解释 chunk、offset、chunk count 和最后一个不完整 chunk，再实现 2-stream 最小正确性版本。不要直接进入性能结论，也不要在没有时间线证据时声称发生重叠。
