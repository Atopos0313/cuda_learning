# CUDA 算子性能与 AI 推理优化 6 周冲刺计划（每周 5 天，共 30 个学习日）

## 0. 路线定位

- **岗位目标**：CUDA 算子性能工程师 / AI 推理算子优化工程师；围绕真实推理 workload，形成实现、验证、profiling、优化和开源源码修改能力。
- **知识组织**：C++、PyTorch、数值精度和 GPU 架构知识随当天算子补充，不单独占学习日。
- **建议投入**：每个学习日 2.5～3 小时。若当天只有 2 小时，优先完成“代码产物 + 验收”，延伸阅读可以延后。
- **工程主线**：写出可运行版本 → 建立正确性基准 → 建立可信 benchmark → 用工具找瓶颈 → 只优化证据指向的部分 → 接入真实框架。
- **学习原则**：理论只学到足以解释代码和性能数据，不以推公式、背架构细节为目标。

### 每日固定节奏

1. 15 分钟：回看前一天的失败案例、性能数据和待办。
2. 20～30 分钟：只学习当天实现需要的概念。
3. 80～100 分钟：编码，至少保留一个朴素版本和一个改进版本。
4. 30～40 分钟：正确性测试、性能测试或 Profiling。
5. 10 分钟：口述瓶颈、修改、证据与适用边界；只保存代码、CSV 和 profiler 等必要证据，不额外写形式化日报。

### 从第一天就使用的工程目录

```text
cuda-engineering/
├─ CMakeLists.txt
├─ cmake/
├─ include/common/          # 错误检查、计时、RAII、数据生成
├─ kernels/                 # .cu 与头文件
├─ apps/                    # 独立可执行样例
├─ tests/                   # 正确性、边界、异常输入
├─ benchmarks/              # 性能测试
├─ projects/llm_kernel_lab/ # Day 11 起的推理算子主项目
├─ scripts/                 # 一键构建、测试、跑 benchmark
└─ reports/                 # CSV/JSON、Nsight 报告、阶段总结
```

### 全程统一验收规则

- **正确性**：GPU 输出始终与 CPU、PyTorch、cuBLAS 或 cuDNN 参考实现比较；浮点运算按 dtype 和运算规模设 `atol/rtol`，不使用“肉眼看起来一样”。
- **边界**：至少覆盖 0、1、非 block 整数倍、质数尺寸、小尺寸和大尺寸；不允许只测整齐尺寸。
- **错误检查**：每次 kernel launch 检查 launch error；关键阶段执行 `compute-sanitizer`，不得有越界、未初始化读取或竞争错误。
- **计时**：先 warm-up；kernel 时间用 CUDA Event；端到端时间用主机时钟并同步；至少记录中位数，不能把首次运行当结果。
- **优化**：任何“更快”的结论都必须有同输入、同精度、同计时范围的基线数据。
- **可复现**：记录 GPU、驱动、CUDA、编译选项、输入形状、dtype、warm-up 次数和迭代次数。

---

## 第一大框：最小 CUDA 工程闭环（第 1～10 天）

### 第一大框通过标准

第 10 天结束时，面对向量、二维矩阵或规约类任务，可以独立完成：建立 CPU 基准、编写 kernel、处理任意尺寸、检查错误、测试边界、可信计时、用 Nsight 找到主要瓶颈，并能解释一次优化为何有效或无效。此时已经具备开始做普通 CUDA 小需求的核心能力。

### 第 1 周：能编译、能运行、能验证

#### Day 1｜建立可重复构建与首个 Kernel 闭环

**工程目标**：从空目录得到一个可一键构建、运行和验证的 CUDA 工程。

**任务**：

1. 确认 GPU、驱动、CUDA Toolkit、编译器和 CMake 可用，记录版本与 GPU Compute Capability。
2. 建立 CMake 工程，分别放置 CUDA kernel、主机调用代码和测试入口。
3. 完成 `vector_add`：申请设备内存、H2D、launch、D2H、释放。
4. 写 CPU 参考实现并逐元素比较，故意制造一次非法 launch，观察错误信息。

**必须掌握知识点**：

- Host/Device 的职责边界；`.cu`、`__global__`、`<<<grid, block>>>` 的含义。
- `threadIdx/blockIdx/blockDim/gridDim` 的最小概念。
- `cudaMalloc/cudaMemcpy/cudaFree` 和 `cudaGetLastError/cudaDeviceSynchronize`。
- CUDA 编译的基本流程；目标架构设置不是“越多越好”。

**可验收标准**：

- 一条命令完成 configure/build，程序退出码为 0。
- 长度为 1、1000、100003 的向量均与 CPU 一致。
- 去掉越界判断后，测试能暴露错误；恢复后正常。
- 错误宏能打印 CUDA API、文件、行号和可读错误文本。

**代码产物**：

- `CMakeLists.txt`
- `kernels/vector_add.cu`
- `apps/vector_add_main.cpp`
- `tests/test_vector_add.cu`
- `include/common/cuda_check.h`

#### Day 2｜索引、网格跨步循环与任意形状

**工程目标**：不依赖“漂亮尺寸”，稳定处理 1D 和 2D 任意输入。

**任务**：

1. 把向量加法改成 grid-stride loop，比较普通单元素索引版本。
2. 实现二维矩阵逐元素缩放或加法，分别用扁平索引和二维 grid/block。
3. 写通用的 `ceil_div` 和 launch 配置辅助函数。
4. 对 0、1、质数尺寸、超大尺寸和非 block 整数倍尺寸做参数化测试。

**必须掌握知识点**：

- 全局线程编号、二维索引映射、边界判断。
- grid-stride loop 的用途：覆盖任意长度、复用线程、避免 grid 大小依赖。
- 行主序数据的线性地址计算。
- block 大小是性能参数，不是正确性条件。

**可验收标准**：

- 1D/2D 两类 kernel 均通过至少 8 组尺寸测试。
- 更换 block size 为 64、128、256 后结果不变。
- 输入为 0 元素时不 launch 非法 grid，并正常返回。
- 能口头从一个 `(blockIdx, threadIdx)` 推导对应数据坐标。

**代码产物**：

- `kernels/elementwise.cu`
- `include/common/launch_config.h`
- `tests/test_indexing.cu`

#### Day 3｜显存生命周期与可靠错误处理

**工程目标**：把裸 CUDA API 封装成不易泄漏、出错能定位的基础设施。

**任务**：

1. 实现移动语义友好的 `DeviceBuffer<T>`，封装分配、释放、大小和指针访问。
2. 统一同步错误与异步 launch 错误检查；明确 Debug 与 Release 检查策略。
3. 用 `compute-sanitizer --tool memcheck` 检查 vector/matrix kernel。
4. 故意做越界写、错误拷贝大小、use-after-free 小实验，保存诊断结论后修复。

**必须掌握知识点**：

- 设备内存的所有权、生命周期、字节数与元素数区别。
- kernel launch 异步，错误可能在后续同步 API 才出现。
- RAII 在 CUDA 资源管理中的价值；不要求复习 C++ 语法本身。
- sanitizer 能发现什么、不能替代什么。

**可验收标准**：

- 正常测试下无显存泄漏、越界或 CUDA API 错误。
- 三类故意注入的错误均能被宏、测试或 sanitizer 捕获。
- `DeviceBuffer` 禁止拷贝、允许移动，重复运行不崩溃。
- 错误信息可以定位到具体调用点，而非只有“程序失败”。

**代码产物**：

- `include/common/device_buffer.h`
- `include/common/cuda_check.h`（增强）
- `tests/test_error_cases.cu`
- `reports/day03_sanitizer_notes.md`

#### Day 4｜正确性测试与可信 Benchmark

**工程目标**：建立后续所有优化共用的测试、计时与数据记录骨架。

**任务**：

1. 实现 CUDA Event 计时器，区分 kernel-only 和 H2D+kernel+D2H 端到端计时。
2. 为 `vector_add` 建立 warm-up、重复迭代、中位数/最小值统计。
3. 计算有效带宽与吞吐量，输出 CSV；同时记录输入规模和硬件软件信息。
4. 添加随机数据、固定种子、误差报告（最大绝对/相对误差和首个错误位置）。

**必须掌握知识点**：

- GPU 异步执行对计时的影响；Event 所在 stream 的语义。
- warm-up、重复采样、同步边界和统计波动。
- kernel-only 与端到端性能回答的是不同问题。
- 浮点误差比较不能只用 `==`。

**可验收标准**：

- 同一 workload 连跑三次，中位数波动可解释；输出不少于 5 个规模点。
- Event 计时和主机计时的适用范围没有混淆。
- CSV 至少含 size、block、warmup、iters、kernel_ms、e2e_ms、GB/s。
- 修改 GPU 输出一个元素后，测试能报告具体索引和误差。

**代码产物**：

- `include/common/cuda_timer.h`
- `include/common/compare.h`
- `benchmarks/bench_vector_add.cu`
- `reports/vector_add_baseline.csv`

#### Day 5｜合并访存：第一个有证据的优化

**工程目标**：通过 A/B kernel 观察连续访存与跨步访存的真实差异。

**任务**：

1. 实现 copy/AXPY 的连续访问版、刻意跨步版和 grid-stride 版。
2. 对多个 block size 和大数据规模做 benchmark，计算有效带宽。
3. 检查指针对齐，在适合的长度上增加 `float4` 向量化 load/store，并保留尾部处理。
4. 写一页结论：哪个版本快、瓶颈是什么、向量化何时无收益。

**必须掌握知识点**：

- warp 发起内存访问、合并访问、对齐、事务浪费。
- 全局内存带宽型 kernel 的判断方式。
- `float4` 的对齐与尾部条件；向量化不等于必然提速。
- block size 调参必须基于数据。

**可验收标准**：

- 三种访问模式结果均正确，非 4 整数倍长度也通过。
- 连续版与跨步版有同条件性能数据，能结合访问模式解释差异。
- 向量化版若没有更快，也能用 profiler/指标给出合理结论，不伪造收益。
- 保存基线与优化后 CSV。

**代码产物**：

- `kernels/memory_access.cu`
- `benchmarks/bench_memory_access.cu`
- `reports/day05_memory_access.csv`
- `reports/day05_conclusion.md`

### 第 2 周：最常用优化、规约、并发与分析工具

#### Day 6｜共享内存与矩阵转置

**工程目标**：用矩阵转置掌握共享内存缓存、同步和 bank conflict。

**任务**：

1. 实现 naive transpose，建立 CPU 参考和不同长宽矩阵测试。
2. 实现 shared-memory tiled transpose。
3. 对 tile 加 padding，比较有/无 bank conflict 版本。
4. 对 copy、naive transpose、tiled transpose 记录有效带宽。

**必须掌握知识点**：

- shared memory 的 block 作用域、生命周期和容量限制。
- `__syncthreads()` 是 block 屏障；所有存活线程必须以一致控制流到达。
- tile、全局合并访存、shared-memory bank conflict。
- 非方阵和边缘 tile 的边界处理。

**可验收标准**：

- 非方阵如 `1003×769` 也正确。
- sanitizer 的 memcheck 和 synccheck 无错误。
- 能从代码指出读写是否合并、padding 消除了哪类冲突。
- 至少保存 naive/tiled/padded 三版数据。

**代码产物**：

- `kernels/transpose.cu`
- `tests/test_transpose.cu`
- `benchmarks/bench_transpose.cu`
- `reports/day06_transpose.csv`

#### Day 7｜规约：从朴素版本到 Warp Shuffle

**工程目标**：实现工程中高频的 sum/max 规约，并正确处理跨 block 合并。

**任务**：

1. 写 CPU sum/max 参考，建立随机值、全负值、极端值和奇数长度测试。
2. 实现 shared-memory block reduction，GPU 输出 partial results，主机或第二级 kernel 合并。
3. 使用 `__shfl_down_sync` 完成 warp reduction，再构造 block reduction。
4. 对多种长度、block size 和两级策略做 benchmark。

**必须掌握知识点**：

- 规约的结合顺序、浮点非结合性与误差累积。
- active mask、warp shuffle、warp 与 block 规约的组合。
- divergent branch、同步位置和非 32 整数倍活跃线程。
- 多 block 输出不能假定执行顺序。

**可验收标准**：

- sum/max 在 0、1、31、32、33、100003 等长度正确。
- max 对全负数输入不以 0 作为错误初值。
- sanitizer 无越界和 race；结果误差阈值有依据。
- 能比较 shared-only 与 shuffle 版，并说明差异来源。

**代码产物**：

- `kernels/reduction.cu`
- `tests/test_reduction.cu`
- `benchmarks/bench_reduction.cu`
- `reports/day07_reduction.csv`

#### Day 8｜Nsight Compute：从指标到代码修改

**工程目标**：学会用证据区分访存、计算、分支和资源瓶颈。

**任务**：

1. 为 copy、transpose、reduction 建立固定可复现的 profiling case。
2. 用 Nsight Compute 采集 kernel 时间、内存吞吐、访存效率、occupancy、warp stall 等关键指标。
3. 选一个指标异常的 kernel，提出一个假设并修改代码验证。
4. 保存 before/after 报告，写出“证据 → 假设 → 修改 → 结果”链条。

**必须掌握知识点**：

- 不需要背全部指标；先看时间、吞吐、访存、occupancy、主要 stall reason。
- achieved occupancy 低不一定是根因，occupancy 高也不保证快。
- profiler 有开销，报告时间不能与普通 benchmark 混用。
- source correlation 和 kernel 过滤的基本用法。

**可验收标准**：

- 能从多个 kernel 中准确选中目标 kernel 并保存报告。
- 报告至少记录 5 个核心指标及其含义。
- 至少完成一次由 profiler 驱动的代码实验，无论最终是否提速。
- 不用“GPU 利用率低”这样模糊结论替代具体指标。

**代码产物**：

- `scripts/profile_ncu.ps1` 或 `scripts/profile_ncu.sh`
- `reports/day08_before.ncu-rep`
- `reports/day08_after.ncu-rep`
- `reports/day08_analysis.md`

#### Day 9｜Stream、Pinned Memory 与拷贝计算重叠

**工程目标**：构造真正的异步数据流水线，并用时间线验证重叠。

**任务**：

1. 写串行版本：H2D → kernel → D2H。
2. 用 pinned host memory、`cudaMemcpyAsync`、多个非默认 stream 和 chunking 写流水线版本。
3. 用 CUDA Event 正确同步每个 stream，处理最后一个不完整 chunk。
4. 用 Nsight Systems 查看 API、Memcpy 和 kernel 时间线，确认是否发生重叠。

**必须掌握知识点**：

- stream 内有序、不同 stream 可并发；默认 stream 语义要明确。
- pageable/pinned host memory 的差异及 pinned memory 成本。
- async API 不代表一定并发；硬件 copy engine、数据依赖与 chunk 大小都会影响。
- Event 依赖比全设备同步更精确。

**可验收标准**：

- 串行与流水线结果完全一致。
- 时间线明确显示重叠；若硬件/工作量不支持，能从时间线说明原因。
- 1、2、4 个 stream 和多个 chunk 大小有对比数据。
- 无过早释放 pinned buffer、无隐式全局同步。

**代码产物**：

- `apps/stream_pipeline.cu`
- `benchmarks/bench_stream_pipeline.cu`
- `scripts/profile_nsys.ps1` 或 `scripts/profile_nsys.sh`
- `reports/day09_timeline.nsys-rep`

#### Day 10｜里程碑 1：核心 Kernel 工具箱 v0.1

**工程目标**：把前 9 天的练习收束成一个别人可以构建、测试和复测的最小工程。

**任务**：

1. 整理 vector/AXPY、transpose、reduction 三类 kernel，统一 API、错误检查和命名。
2. 建立一键 `build → test → benchmark` 流程，测试覆盖非整齐尺寸和错误路径。
3. 为每类 kernel 保存 CPU/朴素 GPU/优化 GPU 的正确性与性能结果。
4. 随机抽一个性能差的版本，用 Nsight Compute 或 Systems 在 30 分钟内定位主要瓶颈。

**必须掌握知识点**：

- CUDA 小工程的模块边界：kernel、host wrapper、测试、benchmark、报告。
- 优化版本必须保留参考版；性能回归必须可复现。
- correctness、kernel latency、end-to-end latency 是三条不同验收线。
- 先定位再优化的工作习惯。

**可验收标准（第 1 里程碑）**：

- 全新构建目录可一次构建成功，全部测试通过。
- `compute-sanitizer` 对核心测试无 memcheck/racecheck/synccheck 问题。
- benchmark 可输出统一 CSV，包含环境与参数。
- 能现场修改一个输入规模并在 30 分钟内得到可信结果和瓶颈判断。
- **达到此标准即获得最常用的 CUDA 核心闭环能力；后续是在扩展算子类型和工程深度。**

**代码产物**：

- `scripts/build_test_bench.*`
- `reports/milestone_01/benchmark.csv`
- `reports/milestone_01/profile.*-rep`
- `reports/milestone_01/README.md`

---

## 第二大框：面向就业的 CUDA 算子与 AI 推理优化（第 11～30 天）

> Day 1～10 保持原计划不变。Day 11 起收缩为 4 周、每周 5 天，围绕一个 `llm_kernel_lab` 主项目推进；原计划 Day 11～40 不再执行。

## 1. 这 20 天只做什么

围绕一个主项目持续推进：

```text
projects/llm_kernel_lab/
├── rmsnorm_swiglu/       内存受限、规约、融合、低精度
├── gemm/                 cuBLAS/CUTLASS、Tensor Core、INT8
├── decode_attention/     online softmax、KV cache、decode workload
├── paged_kv/             从 vLLM 提取的分页寻址实验
├── bench/                统一 workload 与 benchmark
└── profiles/             Nsight 原始报告
```

只追求三类结果：

1. 自己能够从零写出正确的通用 CUDA 版本；
2. 能读懂并修改 Triton、CUTLASS、vLLM 中与当前问题直接相关的实现；
3. 能用正确性、普通 benchmark 和 profiler 证明为什么快或为什么不值得继续优化。

不安排以下内容：

- 不再单独学习 histogram、scan、Thrust、Conv2D、CUDA Graph、NCCL；
- 不做 wheel、安装包、公开 API、通用算子库或“给别人调用”的接口；
- 不设置总结日、里程碑日、发布日；
- 不手写生产级 GEMM、FlashAttention、PTX 或 SASS；
- 不同时学习 Triton、TileLang、TVM、MLIR 多套编译器；主线只选 Triton；
- 不为了覆盖知识点而增加与目标岗位无关的 kernel。

## 2. 自己写、看开源、修改开源的比例

### 50%：自己实现

自己写以下内容：

- RMSNorm CUDA 通用版与优化版；
- fused SwiGLU CUDA 版；
- naive/tiled GEMM 教学版；
- decode attention 正确基线与融合版；
- paged KV cache 最小寻址实验；
- 正确性测试、benchmark 和 profiler workload。

这些代码用于证明你真正理解线程映射、规约、融合、数据布局和数值误差。

### 30%：阅读开源实现

只读与当前算子直接相关的文件：

- Triton 官方 RMSNorm/Softmax/Fused Attention 教程或同类实现；
- CUTLASS profiler、GEMM example、layout/epilogue 相关代码；
- vLLM PagedAttention、RMSNorm、SiluAndMul、benchmark 入口；
- PyTorch SDPA/reference 路径。

不从仓库入口开始“通读整个项目”。每次阅读都从一次实际调用开始，沿着：

```text
Python 调用
→ op 注册或 dispatch
→ wrapper
→ kernel 选择
→ kernel
→ benchmark/test
```

只追踪当前问题需要的链路。

### 20%：修改并吸收开源设计

不整段复制最终实现。只允许三种吸收方式：

1. 把开源实现的 layout、online softmax 或 paged addressing 思路重新写进自己的最小实验；
2. 修改开源 kernel 的 block size、num_warps、tile、dtype 或 shape，预测并验证影响；
3. 给开源项目的 benchmark、测试或 kernel 做一个可以形成独立 Git commit 的小修改。

判断“学会”的标准不是代码跑通，而是能够解释：原版解决什么问题、修改了什么、为什么影响性能、什么 shape 下结论失效。

## 3. 与当前设备适配的版本路线

### 当前设备就是主实验平台

- GPU：RTX 2080 Ti，Turing，Compute Capability 7.5，编译目标固定为 `sm_75`；
- 当前 Windows 驱动可见 CUDA 13.3，原生 CUDA C++、Nsight 和现有 CMake 工程继续直接使用；
- CUDA、CUTLASS、Triton、vLLM 的核心实验优先在这张卡上完成，不默认租 A10/A100；
- Turing 不支持原生 BF16 Tensor Core、TF32、`cp.async`、TMA 和 Hopper warp specialization；这些只在涉及代码分支时理解，不安排伪实验。

### Triton 使用 Turing 兼容线，不追最新版

Triton 官方只支持 Linux，因此在 WSL2 Ubuntu 中建立独立环境。主候选组合固定为：

```text
Python 3.10
PyTorch 2.5.1 + CUDA 12.1 wheel
Triton 3.1.0
GPU target: sm_75
```

这套组合与 vLLM 0.7.x 处于同一代，当时的 vLLM 文档仍明确支持 RTX 20xx/Compute Capability 7.0+。先运行 vector add、reduction、softmax 三个 smoke test；通过后冻结依赖。若个别 Triton 3.1 kernel 触发 Turing 后端限制，再降到 Triton 2.3.x，不切换到非官方 Windows Triton。

当前 Triton 主线最低要求已提高到 Compute Capability 8.0，因此最新版只用于阅读，不作为本机执行环境。依据：[Triton 当前兼容性](https://github.com/triton-lang/triton)、[Triton 2.0 时期兼容性](https://github.com/triton-lang/triton/issues/1057)。

### CUTLASS 与 vLLM 也按 Turing 冻结

- CUTLASS：使用 3.x C++ API，优先从 `3.5.x` 开始，只构建 `75` 架构和当天所需 GEMM；不把 CUTLASS 4.x CuTe DSL/Blackwell 示例纳入主线。
- vLLM：优先使用 `0.7.0～0.7.3` 与 PyTorch 2.5.1/Triton 3.1.0，选择 0.5B～3B 小模型；若完整服务受依赖影响，保留对应 tag 做源码追踪并独立重放目标 kernel，不耗时追新。
- 只有招聘目标明确要求验证 Ampere/Hopper 专有路径时才临时使用云 GPU；它不是本计划前置条件。

依据：[CUTLASS Turing 支持表](https://docs.nvidia.com/cutlass/latest/overview.html)、[vLLM 0.7.0 GPU 要求](https://docs.vllm.ai/en/v0.7.0/getting_started/installation/gpu/)。
## 4. 四周主线

| 周次 | 核心问题 | 最终留下的代码 |
|---|---|---|
| 第 1 周 | 如何优化真实的内存受限融合算子 | RMSNorm + SwiGLU 的 CUDA/Triton 对照 |
| 第 2 周 | GEMM 为什么优先用库，低精度如何进入 Tensor Core | naive/tiled 教学版 + cuBLAS/CUTLASS/INT8 实验 |
| 第 3 周 | Decode Attention 与 KV cache 为什么成为推理瓶颈 | decode attention + paged KV addressing |
| 第 4 周 | 如何在真实推理引擎中定位问题并改动开源代码 | vLLM profile、内部 dispatch、一个干净开源补丁 |

---

# 第 1 周：RMSNorm 与 SwiGLU

## Day 11｜冻结 RTX 2080 Ti 双环境与真实 workload

**工作方式**：保留现有 Windows CUDA 工程，同时建立只服务 Triton/vLLM 的 WSL2 兼容环境；不写新 kernel。

### 当天任务

1. 记录 RTX 2080 Ti、驱动、CUDA 13.3、`sm_75` 和 Windows 基线；运行 Day 10 测试，确认旧工程不回退。
2. 在 WSL2 建立 Python 3.10 + PyTorch 2.5.1/cu121 + Triton 3.1.0，依次跑 vector add、reduction、softmax smoke test；通过后冻结依赖。
3. 建立 `projects/llm_kernel_lab/bench/workloads.json`，固定以下推理 shape：
   - hidden size：`128, 768, 1024, 4096, 8192`；
   - rows/token count：`1, 16, 128, 1024, 4096`；
   - dtype：本地以 FP32/FP16 为主；
   - decode 小 batch 与 prefill 大 token 两类 workload 分开标记。
4. 用 PyTorch 写 RMSNorm 与 SwiGLU reference，保存固定输入与误差阈值。
5. 验证 Nsight Systems、Nsight Compute、compute-sanitizer 在 Linux 可用；处理当前性能计数器权限问题。

### 顺带补充

- Linux 动态库搜索、`ldd`、`PATH`/`LD_LIBRARY_PATH`；
- CMake configure/build/test 的职责；
- PyTorch tensor 的 shape、stride、dtype、device。

### 完成条件

- Windows CUDA 基线不回退，WSL2 中三个 Triton smoke test 通过；
- workload 文件能驱动 PyTorch reference；
- profiler 能采到至少一个 kernel 的 Duration 与 Memory Throughput。

## Day 12｜自己实现 RMSNorm CUDA 通用版

**工作方式**：自己写，不看最终开源 kernel。

### 当天任务

1. 写 FP32 RMSNorm forward：一行一个 block，完成平方和规约、`rsqrt` 和归一化。
2. 使用 FP32 accumulation；明确 epsilon 放置位置。
3. 支持 hidden size 非 2 的幂、空行数、尾部和大 hidden。
4. 与 PyTorch reference 做差分测试。
5. 对 workload 矩阵测 kernel-only 时间，不先调参。

### 顺带补充

- RMSNorm 数学公式；
- 浮点加法不满足结合律；
- block reduction、单位元和尾部线程；
- `rsqrtf` 与 `1 / sqrtf` 的语义和性能边界。

### 完成条件

- workload 全部满足设定容差；
- Sanitizer 无错误；
- 有一份未优化基线 CSV。

## Day 13｜优化 RMSNorm：warp、向量化与 FP16

**工作方式**：自己优化，使用 Nsight 验证。

### 当天任务

1. 把 block reduction 的最后阶段改成 warp shuffle。
2. 为对齐且长度满足条件的输入增加 `float4` 或 half2 读取候选；保留标量 fallback。
3. 增加 FP16 输入/输出、FP32 accumulation。
4. 对不同 hidden size 测 128/256/512 threads，观察寄存器、occupancy、Memory Throughput 和最大 stall。
5. 只保留实际出现稳定收益的 fast path。

### 顺带补充

- alignment 与 vectorized load 前提；
- register pressure 和 occupancy 不是越高越好；
- memory-bound 算子的有效带宽；
- 模板参数与运行时 shape dispatch 的区别。

### 完成条件

- 至少一个高频 shape 相对 Day 12 有稳定收益；
- 不满足对齐/shape 条件时正确 fallback；
- 能用 profiler 指标解释收益或停止优化的原因。

## Day 14｜在 sm_75 上阅读并重写 Triton RMSNorm

**工作方式**：使用冻结的 Triton 3.1.0（必要时降到 2.3.x），在 RTX 2080 Ti 上读同版本实现后独立重写。

### 当天任务

1. 阅读 Triton RMSNorm 或 LayerNorm 实现，只追踪 program id、block size、mask、FP32 accumulation 和 num_warps。
2. 关闭原文件后，按自己的接口重新写 Triton RMSNorm。
3. 与 PyTorch、自己的 CUDA 通用版/优化版比较正确性和性能。
4. 修改 block size、num_warps、是否一次读取完整行，观察不同 hidden size。
5. 查看 Triton 生成的 TTIR/LLVM/PTX 中与 layout、load、reduction 相关的部分，不深入整个编译器。

### 顺带补充

- Triton `program_id`、blocked program、mask；
- JIT specialization、`tl.constexpr`；
- 编译器生成 kernel 与手写 CUDA 的取舍。

### 完成条件

- Triton 版覆盖同一 workload；
- 能解释 Triton 与 CUDA 在至少两个 shape 上的性能差异；
- 不把某一张 GPU 上的结果推广成通用结论。

## Day 15｜SwiGLU 融合：从 PyTorch 链路到 CUDA/Triton

**工作方式**：自己写 CUDA；参考 vLLM/Triton 的 SiluAndMul 后再修改。

### 当天任务

1. 建立未融合 reference：切分 gate/up → SiLU → multiply。
2. 写单 kernel fused SwiGLU CUDA 版，覆盖 FP32/FP16 和非整齐尾部。
3. 写 Triton 版或修改已有 SiluAndMul 教学实现。
4. 比较 eager PyTorch、`torch.compile`、CUDA、Triton 的 kernel 数和端到端时间。
5. 用 Nsight Systems 验证融合是否真的减少中间写回和 launch，而不是只看单 kernel 时间。

### 顺带补充

- SiLU/SwiGLU 在 Transformer MLP 中的位置；
- 融合为什么降低 HBM 流量；
- in-place、alias 和 tensor lifetime 风险。

### 完成条件

- 四条路径语义一致；
- 至少一组真实 shape 有端到端收益；
- 能说明小 shape、对齐失败或编译开销下为什么可能不值得融合。

---

# 第 2 周：GEMM、CUTLASS 与低精度

## Day 16｜建立 LLM GEMM workload 与权威基线

**工作方式**：自己写最小 naive GEMM，生产基线使用 cuBLAS/cuBLASLt。

### 当天任务

1. 从 Transformer 中提取三类 GEMM：
   - prefill：M 较大；
   - decode：M 很小甚至为 1；
   - MLP projection：K/N 较大。
2. 写一个 naive FP32 GEMM，只用于理解 thread→output 映射。
3. 用 PyTorch/cuBLAS 作为正确性与性能基线。
4. 测方阵、长瘦、矮胖和 decode GEMM，记录 FLOP/s、latency 和 layout。
5. 明确 row-major、column-major、leading dimension 和转置标志。

### 顺带补充

- GEMM `M/N/K` 与 Transformer shape 的对应；
- arithmetic intensity；
- decode GEMM 为什么可能更受 latency 和调度影响。

### 完成条件

- 所有 workload 与 cuBLAS reference 对齐；
- 能从 shape 判断是大吞吐 GEMM 还是小 M latency GEMM；
- 不把 naive 版作为后续生产优化目标。

## Day 17｜Shared-memory 与 register-tiled GEMM

**工作方式**：自己实现到“能解释优化”为止，不追赶 cuBLAS。

### 当天任务

1. 写 shared-memory tiled GEMM，支持非整齐 M/N/K。
2. 增加简单 register tiling，让一个线程计算多个输出。
3. 用 NCU 对比 naive/tiled/register-tiled 的 global load、Duration、occupancy、寄存器和 stall。
4. 测至少两组 tile，不使用单一方阵决定配置。
5. 达到“机制已验证”后停止，不继续手写复杂流水或汇编。

### 顺带补充

- 数据复用与算术强度；
- tile、寄存器和 shared memory 的资源交换；
- loop unroll、ILP 与 register spill。

### 完成条件

- tiled 版在合适大矩阵上稳定快于 naive；
- 能解释为什么仍显著落后于 cuBLAS；
- sanitizer 与非整齐 shape 正确。

## Day 18｜CUTLASS：从 profiler 到一次真实修改

**工作方式**：阅读和修改 CUTLASS，不从零写模板库。

### 当天任务

1. 在 Linux 上只构建目标架构和 FP16/FP32 GEMM 子集，避免全库编译。
2. 使用 CUTLASS profiler 跑 Day 16 的 workload，并启用 cuBLAS verification。
3. 阅读一个 device GEMM example，定位 layout、tile shape、warp shape、instruction shape、stages 和 epilogue。
4. 修改一个 tile/stage/alignment 配置，预测后重新 profile。
5. 把 CUTLASS 最佳结果与 cuBLAS、自己的教学 GEMM 放入同一份 workload 数据。

### 顺带补充

- CUTLASS device/threadblock/warp/instruction 分层；
- CuTe layout 只学 shape、stride、composition 的直觉；
- 模板实例化、编译时间和二进制膨胀。

### 完成条件

- 能使用 profiler 搜索并验证 kernel；
- 能指出一个 CUTLASS kernel 的主要 tile/layout 配置；
- 至少完成一次配置修改和 before/after，而不是只运行示例。

## Day 19｜Turing Tensor Core：FP16 与架构边界

**工作方式**：只在 RTX 2080 Ti 上使用 cuBLASLt/CUTLASS 验证真实可用路径。

### 当天任务

1. 运行 FP16 input、FP32 accumulate Tensor Core GEMM，并保留 FP32 CUDA Core 基线。
2. 用 profiler 确认 HMMA/Tensor Core 路径，不根据 API 名称猜测。
3. 测 alignment、K 非整齐、M=1 与大 M，观察 Tensor Core 不占优势的场景。
4. 验证 Turing INT8 Tensor Core 的可用 shape，为 Day 20 建立基线。
5. 只记录影响代码选择的架构差异：Turing 可用 FP16/INT8；BF16/TF32、`cp.async`、TMA、warp specialization 本机不可实测。

### 顺带补充

- storage/compute/accumulation dtype；
- Tensor Core 的 shape/layout/alignment 条件；
- 知道新架构特性不等于当前月必须实测。

### 完成条件

- 至少一个 workload 由 profiler 确认使用 Tensor Core；
- FP16 误差有明确参考和容差；
- 能准确说出 Turing 能做什么、不能做什么，不伪造 Ampere 实验。
## Day 20｜INT8 量化 GEMM：理解精度—性能交换

**工作方式**：自己写量化/reference；GEMM 使用 CUTLASS/cuBLASLt。

### 当天任务

1. 实现 per-tensor 与 per-channel 对称量化的 Python reference：scale、round、clamp、dequant。
2. 构造异常值、不同通道幅度和长 K，比较量化误差。
3. 使用 CUTLASS/cuBLASLt INT8 GEMM，不手写 INT8 Tensor Core mainloop。
4. 比较 FP16 与 INT8 的 kernel、端到端、量化/反量化成本。
5. 对 decode 与 prefill shape 分别判断是否值得量化。

### 顺带补充

- scale/zero point、per-tensor/per-channel；
- accumulator 通常使用 INT32；
- weight-only、W8A8 与校准的区别；
- 理论低精度吞吐不等于端到端收益。

### 完成条件

- 量化误差与性能数据同时存在；
- 至少展示一个 INT8 有收益和一个收益被转换成本抵消的 case；
- 能解释岗位描述中的“量化矩阵乘”究竟包含哪些工程环节。

---

# 第 3 周：Decode Attention 与 Paged KV Cache

## Day 21｜从推理数据流建立 Attention workload

**工作方式**：先建模型与 reference，再读开源算法说明。

### 当天任务

1. 写 PyTorch scaled dot-product attention reference，分开 prefill 与 decode。
2. 固定 batch、heads、head_dim、context length 的 workload：
   - batch：`1, 4, 16`；
   - head_dim：`64, 128`；
   - context：`128, 1024, 4096, 8192`；
   - query length：decode=`1`，prefill 使用较长序列。
3. 手算 QK、scale、softmax、PV 的 shape、FLOPs 和最少内存流量。
4. 阅读 FlashAttention 的 IO-aware/online softmax 思路，以及 vLLM PagedAttention 的 KV 分页动机。
5. 用 PyTorch profiler/Nsight Systems 找出 prefill 与 decode 的主要活动差异。

### 顺带补充

- MHA/GQA/MQA；
- KV cache 容量公式；
- prefill 吞吐与 decode latency；
- causal mask 和数值稳定 softmax。

### 完成条件

- workload 可重复运行且 reference 正确；
- 能说明为什么 decode attention 常常是 memory/latency 问题；
- 不开始实现完整训练版 FlashAttention backward。

## Day 22｜自己实现 unfused Decode Attention 基线

**工作方式**：自己写三个清晰 CUDA 阶段。

### 当天任务

1. 实现 QK dot-product kernel。
2. 复用 reduction 写 stable softmax。
3. 实现 probability × V kernel。
4. 支持 batch/head/context 尾部，FP16 输入、FP32 累加。
5. 与 PyTorch reference 对齐，并测三个 kernel 的时间和中间内存流量。

### 顺带补充

- kernel 间中间 tensor 的 HBM 成本；
- softmax max/sum 两次规约；
- head/block 映射与并行度不足。

### 完成条件

- workload 全部正确；
- 三个阶段可以单独 profile；
- 有明确的未融合性能基线。

## Day 23｜融合 Decode Attention 与 online softmax

**工作方式**：根据论文/开源思想自己写简化版，不复制生产 kernel。

### 当天任务

1. 为一个 query/head 设计 block-level fused kernel。
2. 使用分块 QK 与 online max/sum，避免保存完整 score/probability。
3. 在线更新输出 accumulator，最后归一化并写回。
4. 先限定 head_dim 64/128，其他 shape fallback 到 Day 22。
5. 用普通 benchmark 比较融合前后；用 NCU 判断内存流量、寄存器和 occupancy。

### 顺带补充

- online softmax 更新公式；
- flash/decode attention 的 IO 目标；
- 寄存器 accumulator 与 spill；
- shape specialization 的合理边界。

### 完成条件

- 融合版数值正确；
- 至少一个 decode shape 有稳定收益，或得到可信停止结论；
- 非目标 shape 正确使用 unfused fallback。

## Day 24｜在 sm_75 上实现并修改 Triton Decode Attention

**工作方式**：使用冻结版本中能在 Turing 执行的 attention/softmax 实现，不照搬依赖新架构的最新版教程。

### 当天任务

1. 先运行冻结版本自带的 attention/softmax 测试；遇到 Turing 不支持的路径时关掉对应配置并保留 `sm_75` 安全路径。
2. 沿着 block M/N、head_dim、num_warps、stage、causal 分支理解主循环。
3. 修改一个 tile/num_warps 配置并验证预测。
4. 用相同 workload 比较 PyTorch SDPA、Triton、自己的 CUDA decode kernel。
5. 区分官方 kernel 的 prefill 目标与自己的 decode 目标，避免不公平比较。

### 顺带补充

- Triton `tl.dot`、layout 与 Tensor Core；
- autotune 的搜索空间与成本；
- prefill FlashAttention 和 decode attention 并非同一 workload。

### 完成条件

- 能逐段解释 Triton attention 主循环的数据流；
- 完成一次配置修改和 profiler/benchmark；
- 不声称自己已经实现生产级 FlashAttention。

## Day 25｜从 vLLM 提取 Paged KV 寻址实验

**工作方式**：读 vLLM 源码，吸收一个核心设计到自己的最小项目。

### 当天任务

1. 从 vLLM 的 attention 调用入口追踪到 PagedAttention kernel 与 benchmark/test。
2. 理解 block table、physical block、logical token、head 与 head_size 的地址映射。
3. 在自己的项目中实现两个等价读取实验：contiguous KV 与 paged KV。
4. 控制 context、block size、访问顺序和 batch，比较 locality 与额外寻址成本。
5. 构造错误 block table，确保测试能发现越界/错误映射。

### 顺带补充

- 虚拟内存类比与 KV 分页的区别；
- fragmentation、continuous batching；
- indirect addressing、cache locality 和 coalescing。

### 完成条件

- 不依赖完整 vLLM 也能复现 paged addressing；
- 能从输入 token 定位到物理 KV 地址；
- 有 contiguous/paged 的正确性和性能对比。

---

# 第 4 周：真实推理栈与开源修改

## Day 26｜端到端 Profile vLLM 的 Prefill 与 Decode

**工作方式**：在 WSL2 的 RTX 2080 Ti 上运行冻结的 vLLM 0.7.x；当天先不修改 kernel。

### 当天任务

1. 安装与 PyTorch 2.5.1/Triton 3.1.0 匹配的 vLLM 0.7.x，选择 0.5B～3B 小型 Transformer。
2. 固定模型、prompt、输出长度、batch/concurrency，分别测 prefill 与 decode。
3. 用 vLLM benchmark、PyTorch profiler、Nsight Systems 建立端到端时间线。
4. 记录 top kernel、CPU gap、同步、H2D、通信和显存占用。
5. 从自己的 RMSNorm/SwiGLU/GEMM/attention 中选择与真实热点对应的一项作为后四天目标。

### 顺带补充

- latency、TTFT、TPOT、throughput；
- continuous batching；
- Python/C++/CUDA 调用链和异步时间线。

### 完成条件

- profile 能区分 prefill/decode；
- 目标热点来自时间证据，不是凭岗位关键词选择；
- workload 命令可重复执行。

## Day 27｜建立真实热点的独立 Replay Harness

**工作方式**：从 vLLM 捕获真实 shape，在自己的项目中重放。

### 当天任务

1. 从 Day 26 记录目标 op 的 dtype、shape、stride、batch/context 和调用次数。
2. 在 `projects/llm_kernel_lab` 中构造独立 replay，不启动完整模型。
3. 同时运行 vLLM/框架原版、自己的 CUDA 版和可用的 Triton/CUTLASS 版。
4. 对齐语义与计时边界，找出性能差距。
5. 决定是优化自己的 kernel，还是修改开源 kernel 的一个配置/分支。

### 顺带补充

- microbenchmark 如何保持真实 workload；
- tensor layout/stride、allocator 和 current stream；
- profiler replay 时间与普通性能时间的区别。

### 完成条件

- Replay 结果与真实调用语义一致；
- 能在秒级独立复现目标热点；
- 后续修改不需要每次启动完整模型。

## Day 28｜做一次 Shape-aware 优化与内部选择

**工作方式**：修改自己的 kernel 或开源 kernel，只解决 Day 27 的真实 shape。

### 当天任务

1. 根据 profiler 只选择一个主要变量：访存、tile、warp reduction、num_warps、vectorization 或融合。
2. 实现 before/after，不同时修改多个机制。
3. 在目标 shape 与相邻 shape 上验证，检查优化是否过拟合。
4. 在 benchmark runner 内按 shape/dtype/alignment 选择版本；不制作对外 API。
5. 对不满足 fast-path 的输入保留原版或通用 fallback。

### 顺带补充

- dispatch 条件、代码尺寸和维护成本；
- performance cliff；
- fast path 不是所有 shape 的默认答案。

### 完成条件

- 目标 shape 有稳定收益或得到可信停止结论；
- correctness 与邻近 shape 无退化；
- 选择条件能用数据解释。

## Day 29｜RTX 2080 Ti 的版本、shape 与 fallback 验证

**工作方式**：在同一张 sm_75 GPU 上验证 Windows CUDA、WSL2 CUDA/Triton 和不同 shape 的稳定性。

### 当天任务

1. 对目标 op 固定 `sm_75`，记录 CUDA、Triton、CUTLASS 实际版本和编译目标。
2. 比较相同 workload 在 Windows CUDA 与 WSL2 CUDA/Triton 下的相对结果。
3. 检查 dtype、alignment、tile、shared memory 与寄存器限制，构造 fast path 不适用的 shape。
4. Triton 不支持时回退 CUDA；CUTLASS shape 不匹配时回退 cuBLAS/cuBLASLt。
5. 写出未来迁移 Ampere 时才需重验的清单，不租卡做无目标实验。

### 顺带补充

- `sm_75`、PTX 与实际 cubin；
- 依赖锁定、源码 tag 和复现实验；
- 为什么换环境或 GPU 后不能照搬 block/tile 结论。

### 完成条件

- 本机不会进入 BF16、`cp.async`、TMA 等不支持路径；
- 所有 fast path 都有正确 fallback；
- 能用版本、shape 和 profiler 解释选择。
## Day 30｜完成一个开源质量的小修改

**工作方式**：优先针对冻结的 vLLM/Triton/CUTLASS tag 做一个 Turing 相关小修改，形成独立 commit。

### 可选修改范围

- 增加一个真实缺失的边界测试；
- 修正一个 benchmark 的不公平计时或环境记录；
- 为 Turing shape 增加配置、guard 或 fallback，并提供本机数据；
- 修复一个错误提示、shape guard 或 fallback；
- 给 kernel benchmark 增加一个能复现 performance cliff 的 workload。

### 当天任务

1. 阅读贡献规范和相关测试，不修改无关文件。
2. 写最小复现，证明现状的问题或缺口。
3. 完成代码修改和测试。
4. 跑 before/after 或 failure/pass 证据。
5. 整理为一个范围清晰、可审查的 Git commit；不需要专门写课程总结。

### 顺带补充

- Git diff、commit scope、代码审查；
- 如何用英文写问题、复现步骤和性能结论；
- 开源仓库中的 CI、lint、测试选择。

### 完成条件

- commit 只包含一个主题；
- 修改有测试或 benchmark 证据；
- 能在面试中完整讲清“如何定位、如何理解源码、为什么这样改”。

---

## 5. 每天固定但不单独占天的小知识

这些内容嵌入当天真实代码，不再开独立课程：

### C++

- RAII、move、模板、`constexpr`、指针 const；
- 内存布局、alignment、strict aliasing；
- lambda/function object、编译期与运行时 dispatch；
- 多线程与内存模型只学 benchmark/runner 实际使用部分。

### Python/PyTorch

- tensor shape/stride/view；
- dtype/device/current stream；
- `torch.compile`、custom op、benchmark/profiler；
- 不学习完整训练框架和分布式训练 API。

### 体系结构

- 每次 profiler 遇到再补：memory hierarchy、warp scheduler、register、shared memory、Tensor Core；
- Ampere/Hopper/Blackwell 只学习会改变 kernel 设计的特性；
- 不背 SM 数量、cache 大小等可查参数。

### 面试表达

每天结束前用 10 分钟口述：

```text
这个算子在模型哪里？
瓶颈证据是什么？
我改了什么？
为什么可能更快？
数据是否支持？
在哪些 shape/GPU 上不成立？
```

不额外写日报或长篇总结，答案直接对应当天代码、CSV 和 profiler 报告。

## 6. 一个月后应该能拿出的竞争力证据

完成 20 天后，仓库中应自然留下以下内容，不再另设里程碑：

- RMSNorm：CUDA 通用版、warp/vectorized/FP16 版、Triton 版；
- SwiGLU：PyTorch/CUDA/Triton/torch.compile 公平对比；
- GEMM：naive/tiled 教学版，以及 cuBLAS/CUTLASS/Tensor Core/INT8 选型数据；
- Decode Attention：unfused 基线、online-softmax 融合版、Triton 对照；
- Paged KV：从 vLLM 设计中提取的最小寻址与 locality 实验；
- 一次在 RTX 2080 Ti 上完成的真实 vLLM prefill/decode profile；
- 一个真实 shape 的 replay、优化和 sm_75 环境验证；
- 一个可审查的开源项目 commit。

面试时不需要声称“熟练掌握全部 CUDA 生态”。应该能够基于这些代码回答：

1. 如何建立算子正确性与性能基线；
2. 如何区分 memory-bound、compute-bound 与 launch-bound；
3. CUDA、Triton、cuBLAS/CUTLASS 分别何时使用；
4. RMSNorm、SwiGLU、GEMM、Decode Attention 在推理链路中的瓶颈；
5. 为什么同一优化在不同 shape、dtype 和 GPU 上结果不同；
6. 如何阅读并修改 vLLM/Triton/CUTLASS 的真实代码。

这比完成大量互不相关的教学 kernel 更接近目标岗位的实际工作。
