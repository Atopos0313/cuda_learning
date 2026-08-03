# Day 9｜Stream、Pinned Memory 与拷贝计算重叠

> 2026-08-03 学习进度：**Day 9 已完成**。串行与异步正确性、chunking、1/2/4-stream Release benchmark 以及 Nsight Systems 时间线证据均已完成并归档。时间线证明本次工作负载没有发生 copy/compute 重叠，只观察到少量 concurrent kernel 重叠。

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

## 小阶段 3｜Chunking 与 2-stream 正确性（已完成）

### 本阶段目标与完成条件

本阶段把完整数组切成多个连续小段，并把这些小段轮流提交给 2 条 stream。只有同时满足以下条件，才算完成：

- 能手算每个 chunk 的 `offset`、实际元素数和所属 stream；
- 最后一个不完整 chunk 不越界，也不遗漏；
- 两条 stream 使用互不冲突的 device buffer；
- 每个 chunk 的 H2D、kernel、D2H 位于同一条 stream；
- CPU 等待两条 stream 的完成点后才检查输出；
- 完整的 `N = 1,000,003` 输出通过同一 CPU reference；
- 保存实际输出并记录已验证结论。

### Chunk 是什么

**Chunk（数据块）** 是从完整数组中切出的一个连续区间。没有 chunking 时，一次处理全部 `N` 个元素：

```text
[0 ........................................ N-1]
```

有了 chunking 后，数组会被切成多个连续且不重叠的区间：

```text
[chunk 0][chunk 1][chunk 2]...[最后一个 chunk]
```

切块不是改变计算公式。每个元素仍执行 `output[i] = input[i] * 2 + 1`；变化的只是数据分批进入 GPU，从而让不同批次有机会在不同 stream 中形成传输与计算重叠。

### 四个必须分清的量

设完整元素数为 `N`，期望的块大小为 `chunk_size`，当前块编号为 `chunk_id`。

1. `chunk_count`：总共需要多少个 chunk。

```cpp
chunk_count = (N + chunk_size - 1) / chunk_size;
```

2. `offset`：当前 chunk 在完整数组中的起始位置。

```cpp
offset = chunk_id * chunk_size;
```

3. `remaining`：从 `offset` 开始还剩多少个元素。

```cpp
remaining = N - offset;
```

4. `current_count`：当前 chunk 实际处理多少个元素。

```cpp
current_count = min(chunk_size, remaining);
```

前三个完整 chunk 的 `current_count` 通常等于 `chunk_size`；最后一个 chunk 可能更短，所以必须重新计算。

### 手算一个尾块

假设：

```text
N = 23
chunk_size = 8
stream_count = 2
```

则：

| chunk_id | offset | current_count | stream_id |
|---:|---:|---:|---:|
| 0 | 0 | 8 | 0 |
| 1 | 8 | 8 | 1 |
| 2 | 16 | 7 | 0 |

最后一个 chunk 覆盖全局下标 `16～22`，共 7 个元素。若错误地继续按 8 个元素传输，程序就会访问下标 23，越过数组末尾。

### 两条 Stream 不等于只有两个 Chunk

Stream 是可重复使用的有序任务队列。若 chunk 多于 stream，可用轮转分配：

```cpp
stream_id = chunk_id % stream_count;
```

例如 5 个 chunk、2 条 stream 的分配是：

```text
chunk 0 → stream 0
chunk 1 → stream 1
chunk 2 → stream 0
chunk 3 → stream 1
chunk 4 → stream 0
```

同一 stream 内仍严格有序，所以 `chunk 2` 提交到 stream 0 后，会排在 `chunk 0` 的 D2H 后面。不同 stream 之间没有这种自动顺序。

### Host 与 Device 地址怎样对应

Host 端保存的是完整 pinned 数组，所以当前 chunk 从 `h_input_pinned + offset` 开始，结果写回 `h_output_pinned + offset`：

```text
完整 Host 输入  + offset → 当前 chunk 的 Host 起点
当前 stream 的 DeviceBuffer → 当前 chunk 在 GPU 上的局部起点
完整 Host 输出  + offset ← 当前 chunk 的结果写回位置
```

Device 端建议为每条 stream 准备自己的输入、输出 buffer。原因是两条 stream 可能并发访问；若共用同一 device buffer，stream 1 的 H2D 可能覆盖 stream 0 尚未计算完的数据。

Kernel 内部仍使用局部下标 `0～current_count-1`。全局位置由 Host 指针的 `offset` 负责，kernel 不需要知道完整数组中的全局 offset。

### 尾块必须同时修改三个量

最后一个 chunk 不能只修改 kernel 参数，还必须让以下三个量全部使用 `current_count`：

```text
传输字节数：current_count * sizeof(float)
kernel 的 n：current_count
grid 大小：ceil(current_count / threads_per_block)
```

常见错误是 D2H 使用了正确尾块字节数，却仍把 `chunk_size` 传给 kernel；这样 kernel 仍可能读写 device buffer 中不属于本次尾块的旧数据。

### 2-stream 数据流

本阶段随后要实现的最小正确性模型是：

```text
Stream 0：H2D(chunk 0) → kernel(chunk 0) → D2H(chunk 0)
Stream 1：H2D(chunk 1) → kernel(chunk 1) → D2H(chunk 1)
```

若还有 chunk 2，它可以继续排到 stream 0 的 D2H(chunk 0) 后面。当前阶段只验证数据切分、队列归属和最终结果；是否真的发生重叠，仍要等 Nsight Systems 时间线。

### 分块配置与手算检查

固定：

```text
N = 1,000,003
chunk_size = 262,144
stream_count = 2
threads_per_block = 256
```

本阶段已通过手算与程序输出共同验证：

| chunk_id | offset | current_count | current_bytes | chunk_grid | stream_id |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 262144 | 1048576 | 1024 | 0 |
| 1 | 262144 | 262144 | 1048576 | 1024 | 1 |
| 2 | 524288 | 262144 | 1048576 | 1024 | 0 |
| 3 | 786432 | 213571 | 854284 | 835 | 1 |

尾块满足 `786432 + 213571 = 1000003`，恰好覆盖到完整数组末尾，没有遗漏或越界。

### 实际代码数据流

每条 stream 使用独立的输入、输出 DeviceBuffer；Host 端仍保留完整 pinned 数组：

```text
h_input_pinned + offset
→ chunk_inputs[stream_id]
→ simpleKernel（局部下标 0～current_count-1）
→ chunk_outputs[stream_id]
→ h_output_pinned + offset
```

同一 stream 内按 `H2D → kernel → D2H` 排队。chunk 0、2 进入 stream 0，chunk 1、3 进入 stream 1；同一 stream 再次使用自己的 DeviceBuffer 前，前一个 chunk 的 D2H 已按队列顺序排在前面。

### 构建与实际输出

Debug 配置重新构建并运行成功。原始证据保存在 [2-stream 正确性输出](results/2stream_correctness_debug.txt)。关键输出为：

```text
mode=serial_pageable, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
mode=serial_pinned, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
mode=async_pinned, streams=1, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
mode=2stream_copy, streams=2, size=1000003, correct=PASS, max_abs_error=0, max_rel_error=0
```

`mode=2stream_copy` 是学习者在代码中选择并保留的模式名。该路径并非只做 copy，实际执行的是分块的 `H2D → simpleKernel → D2H`，并与 CPU `reference` 比较。

### 已验证结论

- 2 条 non-blocking stream 和两组独立 DeviceBuffer 创建、使用与销毁成功；
- 4 个 chunk 完整覆盖 `N = 1,000,003`，最后一个不完整 chunk 正确使用 213571 个元素；
- 每个 chunk 的传输字节数、kernel 元素数和 grid 都由 `current_count` 计算；
- H2D、kernel 与 D2H 提交到同一条对应 stream；
- CPU 在两条 stream 都同步完成后才读取完整 Host 输出；
- 2-stream 完整输出与 CPU reference 完全一致，最大绝对误差和相对误差均为 0；
- 程序最终退出状态包含 2-stream 比较结果，不会在该路径 FAIL 时假返回成功。

### 结论边界

- 本阶段是 Debug 正确性实验，没有预热、重复计时或固定性能计时范围；
- 两条 stream 提供潜在并发机会，但当前证据不能证明 H2D、kernel 与 D2H 实际重叠；
- `cudaStreamSynchronize` 适合当前正确性收口，正式 benchmark 需要设计统一的开始/结束计时边界；
- 尚未验证 1/2/4 streams 和不同 chunk size 的性能关系。

### 下一小阶段

小阶段 4 直接沿用仓库既有 benchmark 规范：Release 构建、20 轮预热、50 轮正式测量并报告中位数。教学只展开 CUDA 特有的多 stream 完成边界；没有普通 benchmark 和 Nsight Systems 时间线之前，不宣称 2-stream 更快或已经发生重叠。

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
- 尚未生成 `benchmarks/bench_stream_pipeline.cu` 和 `reports/day09_timeline.nsys-rep`；
- 尚未使用 Nsight Systems 证明传输与计算是否重叠；
- 当前 pinned、stream 和 event 仍为手动资源管理，异常路径 RAII 化待完成。

### 当时的下次继续入口

小阶段 3 已开始。先完成文中的 chunk 手算检查，再实现 2 条 stream 和两组独立 device buffer；不要直接进入性能结论，也不要在没有时间线证据时声称发生重叠。

## 2026-08-03 续学记录

### 本次已完成

- 从零建立 chunk、offset、remaining、current count 和尾块模型；
- 手算并用程序验证 4 个 chunk 的覆盖范围、字节数、grid 和 stream 轮转分配；
- 创建 2 条 non-blocking stream 和两组独立输入、输出 DeviceBuffer；
- 完成分块 `H2D → simpleKernel → D2H` 计算链；
- 通过两条 stream 的完成同步保护 Host 读取；
- 2-stream 完整结果通过 CPU reference，最大绝对误差与相对误差均为 0；
- 保存 Debug 正确性原始输出，并记录 `mode=2stream_copy` 的实际语义。

### 后续完成情况

- 已完成 1/2/4 streams 与多个 chunk size 的 Release benchmark；
- benchmark 已实现每条活动 stream 的 completion event，并等待全部 event 形成统一完成边界；
- 已生成 `benchmarks/bench_stream_pipeline.cu` 和 `reports/day09_timeline.nsys-rep`；
- 按学习偏好保留手动输入 `nsys profile` / `nsys stats` 的流程，不使用一键 profiling 脚本；
- 已使用 Nsight Systems 确认本次工作负载没有 copy/compute overlap，但存在少量 concurrent kernel；
- pinned、stream 和 event 已提取为公共 RAII 工具。

### 当前继续入口

小阶段 4 和小阶段 5 均已完成。Release benchmark 加速不能归因于 copy/compute overlap；下一步进入 Day 10 工具箱 v0.1 收束。

## 小阶段 4：1/2/4-stream Release benchmark（已完成）

### 实验配置

- `N = 16,777,219`，`block = 256`；
- stream 数：`1 / 2 / 4`；
- chunk size：`65,536 / 262,144 / 1,048,576` 个元素；
- Release 构建，20 轮预热、50 轮正式测量，报告端到端中位数；
- 端到端范围覆盖全部 chunk 的 pinned H2D、`simpleKernel` 和 pinned D2H；
- 每个活动 stream 在自己的最后一个 D2H 后记录 completion event，Host 等待所有 event，不使用 `cudaDeviceSynchronize()`。

### 多 stream 的统一完成条件

每条 stream 只保证自己队列内部有序。因此，stream 0 的 completion event 完成，只能证明 stream 0 的 chunk 已完成，不能证明 stream 1～3 已完成。本 benchmark 在提交全部 chunk 后，分别在每条活动 stream 尾部记录 event：

```text
stream 0: ... -> D2H(last chunk on stream 0) -> event 0
stream 1: ... -> D2H(last chunk on stream 1) -> event 1
stream 2: ... -> D2H(last chunk on stream 2) -> event 2
stream 3: ... -> D2H(last chunk on stream 3) -> event 3
Host: wait(event 0, event 1, event 2, event 3)
```

只有所有活动 event 都完成，才能安全读取完整 Host 输出、开始下一轮并复用每条 stream 的 device buffer。event 使用 `cudaEventDisableTiming`，只表达完成边界；端到端测量由 Host 计时器包围“提交全部任务 + 等待全部 event”。

### 实测结果

原始 CSV：[stream_pipeline_release.csv](results/stream_pipeline_release.csv)

| chunk size | chunks | 1 stream | 2 streams | 4 streams | 最快配置 |
|---:|---:|---:|---:|---:|---:|
| 65,536 | 257 | 69.615000 ms | 34.901750 ms | 22.573100 ms | 4 streams，3.084x |
| 262,144 | 65 | 21.297850 ms | 16.429450 ms | 14.338550 ms | 4 streams，1.485x |
| 1,048,576 | 17 | 14.759050 ms | 12.832200 ms | 12.023850 ms | 4 streams，1.227x |

9 组结果均为 `PASS`，最大绝对误差和最大相对误差均为 0。`speedup_vs_1stream` 只与相同 chunk size 的 1-stream 结果比较。

### 已验证结论与边界

- 已验证：1/2/4-stream 三种调度都完整覆盖不规则尾块，并与 CPU reference 一致；
- 已验证：在本机、本次固定配置下，增加 stream 数降低了端到端中位时间；
- 已验证：本次候选配置中，`chunk_size=1,048,576`、`streams=4` 的时间最低，为 `12.023850 ms`；
- 初步判断：`chunk_size=65,536` 的 1-stream 路径包含较多小 chunk 提交与顺序执行成本；
- 已验证：benchmark 时间下降本身不能证明 H2D、kernel、D2H 已实际重叠；Nsight Systems 进一步确认本配置没有 copy/compute overlap。

### 本阶段后的验证节点

小阶段 5 已完成：生成 Nsight Systems 时间线，直接观察不同 stream 上的 H2D、`simpleKernel`、D2H 是否在时间轴上重叠，并把 `.nsys-rep`、CSV、截图与分析结论归档。

### 代码整理记录

完成小阶段 4 后，将与流水线算法无关的通用实现提取到 `include/common`：

- `pinned_buffer.h`：自动分配和释放 pinned host memory；
- `cuda_stream.h`：自动创建和销毁 non-blocking stream，并提供 `get()`、`synchronize()`；
- `cuda_event.h`：自动管理禁用计时的 completion event，并提供 `record()`、`synchronize()`；
- `benchmark_stats.h`：提供所有 benchmark 共用的 `benchmark_median()`。

`apps/stream_pipeline.cu` 和 `benchmarks/bench_stream_pipeline.cu` 已改为调用上述工具。原先四个 benchmark 中重复的中位数实现也已删除并统一调用公共函数。重构后的 Release 全量构建通过；正确性应用四条路径全部 `PASS`，9 组流水线 benchmark 全部 `PASS`。

## 小阶段 5：Nsight Systems 时间线（已完成）

### Nsight Systems 解决什么问题

普通 benchmark 只能观察整轮时间是否变短，不能说明 H2D、kernel、D2H 在 GPU 上如何排列。Nsight Systems 为每次 CUDA API 和 GPU 活动记录开始时间、持续时间、stream ID，并把它们放到同一条横向时间轴。只有两个活动区间真实相交，才能认定它们发生了重叠；源码中存在多个 stream 不是重叠证据。

### 采集配置与手动命令

- Release 构建；
- `N = 16,777,219`，`block = 256`；
- `streams = 4`，`chunk_size = 1,048,576`，共 17 个 chunk；
- 资源创建和一次预热位于采集范围外；
- `cudaProfilerStart()` 到 `cudaProfilerStop()` 之间只采集一轮流水线；
- `cudaProfilerStop()` 之前由每条 stream 的 completion event 保证全部 GPU 工作完成。

普通 benchmark 使用 20 轮预热和 50 轮测量；`--profile` 是独立路径，Nsight 不是从 50 轮中挑选一轮。为熟悉工具，本阶段保留手动命令，不提供一键脚本：

```powershell
nsys profile `
  --trace cuda `
  --capture-range cudaProfilerApi `
  --capture-range-end stop `
  --force-overwrite true `
  --output .\reports\day09_timeline `
  .\build\windows-ninja\Release\bench_stream_pipeline.exe `
  --profile

nsys stats --report cuda_gpu_sum .\reports\day09_timeline.nsys-rep
nsys stats --report cuda_api_sum .\reports\day09_timeline.nsys-rep
```

归档产物：

- [Nsight Systems 时间线报告](../../reports/day09_timeline.nsys-rep)
- [CUDA GPU 活动明细 CSV](results/day09_cuda_gpu_trace.csv)
- [Nsight Systems 专题分析](day09_nsys_stream_pipeline_analysis.md)

### 时间线与统计结论

![完整 CUDA HW 时间线](images/05_full_cuda_hw_timeline.png)

| GPU 活动 | 占比 | 次数 | 总时间 | 中位时间 |
|---|---:|---:|---:|---:|
| Pinned H2D | 50.0% | 17 | 5.827001 ms | 0.356956 ms |
| Pinned D2H | 47.6% | 17 | 5.550110 ms | 0.339163 ms |
| `simpleKernel` | 2.4% | 17 | 0.280763 ms | 0.017695 ms |

传输占 GPU 活动时间的 97.6%，累计传输时间约为累计 kernel 时间的 40.5 倍。逐条比较最终 CSV 的 `[start, start + duration)` 区间：

| 重叠类型 | 相交次数 | 最大相交时间 |
|---|---:|---:|
| H2D 与 kernel | 0 | 0 ns |
| kernel 与 D2H | 0 | 0 ns |
| H2D 与 D2H | 0 | 0 ns |
| 不同 stream 的 kernel 与 kernel | 12 对 | 3,040 ns |

实际 GPU 排列呈现为：

```text
一批 H2D -> 一批短 kernel -> 一批 D2H -> 下一批 H2D -> ...
```

因此准确结论是：**本配置没有 copy-compute overlap，但存在少量 concurrent kernel。** Pinned memory 为可靠异步 DMA 提供必要条件，却不保证一定发生硬件重叠。

### CUDA API 与同步审计

- 34 次 `cudaMemcpyAsync` 对应 17 次 H2D 和 17 次 D2H；API duration 只是 CPU 提交时间，不是 GPU copy 时间；
- 17 次 `cudaLaunchKernel` 对应 17 个 chunk；
- 4 次 `cudaEventRecord` 和 4 次 `cudaEventSynchronize` 对应 4 条 stream 的统一完成边界；
- 没有 `cudaDeviceSynchronize` 或 `cudaStreamSynchronize`；
- `cudaProfilerStart` 的长耗时属于 profiler 初始化，不属于流水线工作负载。

### Day 9 专题文档与截图

- [Nsight Systems 流水线专题分析](day09_nsys_stream_pipeline_analysis.md)
- [C++ / CUDA 代码阅读问答](day09_cpp_cuda_notes.md)
- [Day 9 报告索引](../../reports/day09_analysis.md)
- [时间线全局视图](images/01_timeline_overview.png)
- [Stream 13 H2D 详情](images/02_h2d_stream13_detail.png)
- [Stream 13 kernel 详情](images/03_kernel_stream13_detail.png)
- [Stream 14 H2D 详情](images/04_h2d_stream14_detail.png)
- [完整 CUDA HW 时间线](images/05_full_cuda_hw_timeline.png)

### Day 9 收尾

Day 9 的验收项已完成：串行与流水线结果一致；1/2/4 streams 和多个 chunk size 有 Release 数据；每条 stream 使用明确 completion event；没有全设备同步或过早释放 pinned buffer；Nsight Systems 明确证明本工作负载没有形成 copy/compute overlap，且时间主要花在数据传输。

下一学习节点是 Day 10：把前 9 天的 kernel、公共 CUDA 资源工具、测试、benchmark 和文档收束为核心 Kernel 工具箱 v0.1。
