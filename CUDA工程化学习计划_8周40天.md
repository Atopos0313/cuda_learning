# CUDA 工程化 8 周冲刺计划（每周 5 天，共 40 个学习日）

## 0. 路线定位

- **目标**：尽快获得“能独立写 CUDA Kernel、验证正确性、定位性能问题、接入 PyTorch、形成可交付工程”的能力。
- **不纳入本计划**：C++ 语法、STL、面向对象等基础；这些继续用零散时间补。
- **建议投入**：每个学习日 2.5～3 小时。若当天只有 2 小时，优先完成“代码产物 + 验收”，延伸阅读可以延后。
- **工程主线**：写出可运行版本 → 建立正确性基准 → 建立可信 benchmark → 用工具找瓶颈 → 只优化证据指向的部分 → 接入真实框架。
- **学习原则**：理论只学到足以解释代码和性能数据，不以推公式、背架构细节为目标。

### 每日固定节奏

1. 15 分钟：回看前一天的失败案例、性能数据和待办。
2. 20～30 分钟：只学习当天实现需要的概念。
3. 80～100 分钟：编码，至少保留一个朴素版本和一个改进版本。
4. 30～40 分钟：正确性测试、性能测试或 Profiling。
5. 10 分钟：在 `reports/learning_log.md` 记录结论、数据、错误与下一步。

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
├─ pytorch_ext/             # 第 5 周开始
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

## 第二大框：并行原语与常见深度学习算子（第 11～20 天）

### 第二大框通过标准

第 20 天结束时，能够判断任务属于 elementwise、规约、scan、histogram、GEMM、卷积还是 softmax，并优先判断应调用成熟库还是自写 kernel；若自写，可以给出正确基线、至少一版合理优化和与权威参考实现的对比。

### 第 3 周：常用并行原语与数据布局

#### Day 11｜原子操作与直方图

**工程目标**：理解并解决大量线程更新少量输出位置时的竞争与争用。

**任务**：

1. 实现 256-bin 全局原子直方图，使用 CPU 结果校验。
2. 实现每 block 私有 shared histogram，再归并到 global。
3. 构造均匀分布与热点分布输入，比较 contention 的影响。
4. 使用 racecheck 验证去掉原子操作后的错误，再恢复正确实现。

**必须掌握知识点**：

- 数据竞争、atomic 的正确性语义和支持的数据类型。
- 原子操作正确不代表高性能；冲突分布决定代价。
- block 私有化、分层聚合是减少全局争用的通用方法。
- shared-memory atomics 与 global atomics 的取舍。

**可验收标准**：

- 均匀、单热点、多热点、任意长度输入均正确。
- 非原子错误版本能被测试或 racecheck 发现。
- 两种实现有不同数据分布下的性能曲线。
- 能说明何时私有化会因 shared memory 或归并成本得不偿失。

**代码产物**：

- `kernels/histogram.cu`
- `tests/test_histogram.cu`
- `benchmarks/bench_histogram.cu`
- `reports/day11_histogram.csv`

#### Day 12｜Prefix Scan：实现局部、组合全局

**工程目标**：掌握 scan 的工程分解，同时知道何时不应重复造轮子。

**任务**：

1. 实现单 block exclusive scan，测试非 2 的幂长度。
2. 将大数组分解为 block scan、block sums scan、uniform add 三阶段。
3. 用 CUB `DeviceScan` 作为正确性和性能参考。
4. 比较自写版与 CUB 在多个规模下的时间和临时内存需求。

**必须掌握知识点**：

- inclusive/exclusive scan 的接口语义。
- 多 block 算法需要额外阶段，不能跨 block 直接同步。
- work-efficient scan 的目标，只学到能看懂实现与复杂度。
- CUB 临时存储查询与复用。

**可验收标准**：

- 0、1、非 2 的幂和百万级长度均正确。
- 处理 block sums 的递归或分层逻辑没有固定规模假设。
- 对 CUB 的临时存储只分配一次并在重复迭代中复用。
- 能明确说明项目中默认选 CUB 还是自写版，以及理由。

**代码产物**：

- `kernels/scan.cu`
- `tests/test_scan.cu`
- `benchmarks/bench_scan_vs_cub.cu`
- `reports/day12_scan.csv`

#### Day 13｜Thrust/CUB：库优先的工程决策

**工程目标**：用成熟并行原语快速拼出真实数据处理流程。

**任务**：

1. 用 Thrust 或 CUB 完成 filter/compact：predicate → scan/select → 输出。
2. 实现 sort-by-key + reduce-by-key 的小型聚合任务。
3. 管理临时存储和 stream，避免每轮动态分配与无意同步。
4. 与自写 scan/reduction 比较开发成本、性能和可维护性。

**必须掌握知识点**：

- Thrust 高层容器/算法与 CUB 低层原语的定位。
- 临时存储查询、allocator 复用、指定 stream。
- iterator/transform 的融合机会；警惕隐藏同步和分配。
- “库能做就先用库；profile 证明必要后才替换”。

**可验收标准**：

- compact 输出长度和内容与 CPU 参考一致。
- 聚合任务能处理重复 key、空输入和随机 key。
- profiler 中循环内无不必要的 `cudaMalloc/cudaFree`。
- 写出一张自写/CUB/Thrust 选型表。

**代码产物**：

- `apps/primitives_pipeline.cu`
- `benchmarks/bench_library_primitives.cu`
- `reports/day13_selection_guide.md`

#### Day 14｜数据布局、对齐与 Layout Transform

**工程目标**：能从访问方式反推 AoS、SoA、NCHW、NHWC 等布局选择。

**任务**：

1. 构造粒子/特征数据的 AoS 与 SoA 两种结构，执行同一字段运算。
2. 实现 AoS↔SoA 或 NCHW↔NHWC layout transform。
3. 检查 pitch/stride/contiguous 条件，支持带 stride 的输入或明确拒绝。
4. 用 Nsight Compute 对比 global load/store 效率和有效带宽。

**必须掌握知识点**：

- logical shape、physical layout、stride 的区别。
- 对齐、连续访问和 vectorized memory access 的条件。
- layout transform 本身有成本，需在后续收益中摊销。
- 接口必须声明是否要求 contiguous。

**可验收标准**：

- 往返 transform 后数据一致，奇数维度也通过。
- AoS/SoA 至少一个实际访问任务有对比数据。
- 能根据下游访问模式选择布局，而不是默认某一种永远更快。
- 对不支持的 stride 输入给出清晰错误，不静默算错。

**代码产物**：

- `kernels/layout_transform.cu`
- `tests/test_layout_transform.cu`
- `benchmarks/bench_layout.cu`
- `reports/day14_layout.csv`

#### Day 15｜浮点精度、FP16/BF16 与数值契约

**工程目标**：让“更低精度”成为经过测试的工程选项，而不是简单替换类型。

**任务**：

1. 为 AXPY/reduction 增加 FP16，硬件支持时增加 BF16 输入版本。
2. 比较低精度累加与 FP32 累加的误差和性能。
3. 构造大值、小值、相消、长规约等数值压力测试。
4. 为每个 dtype 制定并记录误差容限与累加策略。

**必须掌握知识点**：

- storage dtype、compute dtype、accumulation dtype 可以不同。
- overflow、underflow、舍入和长规约误差。
- half2 等向量化的对齐与偶数长度尾部处理。
- 性能提升受硬件、算子类型和转换开销影响。

**可验收标准**：

- FP32/FP16（及可用时 BF16）均有明确正确性阈值。
- 压力测试能显示不同累加策略的误差差异。
- 任意长度含奇数尾部正确。
- 不以放宽到失去意义的容差掩盖实现错误。

**代码产物**：

- `kernels/mixed_precision.cu`
- `tests/test_numerics.cu`
- `benchmarks/bench_precision.cu`
- `reports/day15_numerical_contract.md`

### 第 4 周：常见深度学习算子

#### Day 16｜GEMM 基线与 cuBLAS 参考

**工程目标**：建立矩阵乘法的正确性、布局和性能比较框架。

**任务**：

1. 实现 naive SGEMM，每个线程计算一个输出元素。
2. 用 CPU 小规模结果和 cuBLAS 大规模结果作为双参考。
3. 处理 M/N/K 非 tile 整数倍、转置标志和 `alpha/beta`。
4. 输出不同形状下 latency、GFLOP/s 与误差。

**必须掌握知识点**：

- GEMM 接口 `C = alpha·op(A)·op(B) + beta·C`。
- 行主序/列主序与 leading dimension；不要用“转置一下似乎对了”糊弄。
- GEMM 计算量和 GFLOP/s 的计算方式。
- 自写 GEMM 的目的主要是理解优化，生产默认优先 cuBLAS/cuBLASLt。

**可验收标准**：

- 方阵、长瘦、矮胖和非整齐 K 均正确。
- 至少 FP32 下与 cuBLAS 对齐，并记录容差。
- benchmark 不把 handle 创建和首次初始化算进 steady-state kernel 时间。
- 能准确解释每个矩阵的布局与 leading dimension。

**代码产物**：

- `kernels/gemm_naive.cu`
- `include/reference/cublas_gemm.h`
- `tests/test_gemm.cu`
- `benchmarks/bench_gemm.cu`

#### Day 17｜Tiled GEMM：共享内存与寄存器复用

**工程目标**：在不追求手写库级极限的前提下，完成一次完整 GEMM 优化。

**任务**：

1. 实现 shared-memory tiled GEMM，边缘 tile 做零填充。
2. 增加每线程多个输出或简单 register tiling 版本。
3. 用 Nsight Compute 检查 global load、shared memory、occupancy 和 stall。
4. 与 naive 和 cuBLAS 比较至少 6 个典型形状。

**必须掌握知识点**：

- arithmetic intensity、数据复用、tile 尺寸的资源代价。
- shared-memory tile、寄存器累加和同步点。
- register pressure、occupancy 与 ILP 的权衡。
- 自写版显著落后于 cuBLAS 是正常结果，重点是能解释差距。

**可验收标准**：

- 任意 M/N/K 正确，sanitizer 无错。
- tiled 版相对 naive 在适合的大矩阵上有稳定改进，或能解释未改进原因。
- 至少测试两种 tile/block 配置，不凭感觉选参数。
- 报告中明确库版、自写版的适用边界。

**代码产物**：

- `kernels/gemm_tiled.cu`
- `benchmarks/bench_gemm_variants.cu`
- `reports/day17_gemm.ncu-rep`
- `reports/day17_gemm_analysis.md`

#### Day 18｜二维卷积：先正确，再决定调用库还是融合

**工程目标**：掌握卷积的 shape/layout/边界，并建立 cuDNN 或框架参考。

**任务**：

1. 实现简化版 NCHW direct Conv2D，支持 stride、padding，先限定 dilation=1、groups=1。
2. 用 PyTorch 或 cuDNN 输出校验，覆盖非方形输入和 kernel。
3. 增加 constant/read-only 权重或 shared-memory input tile 实验版。
4. 比较 direct、自写优化和库实现，记录端到端与 kernel 时间。

**必须掌握知识点**：

- N/C/H/W、K/R/S、输出 shape 公式和边界。
- 输入、权重、输出的线性索引；stride/padding 的语义。
- direct、im2col+GEMM、库算法的工程取舍。
- 自写卷积通常只在形状固定、融合或特殊布局时有价值。

**可验收标准**：

- 至少 10 组 shape/stride/padding 与参考一致。
- 无越界读取；padding 区域逻辑清楚。
- 能说明自写版与库差距来自哪里，以及是否值得继续优化。
- 接口对暂不支持的 groups/dilation 明确报错。

**代码产物**：

- `kernels/conv2d_direct.cu`
- `tests/test_conv2d.cu`
- `benchmarks/bench_conv2d.cu`
- `reports/day18_conv2d.csv`

#### Day 19｜Stable Softmax 与融合思维

**工程目标**：实现数值稳定、适配常见行宽的 row-wise softmax。

**任务**：

1. 写 CPU/PyTorch stable softmax 参考：先求 max，再 exp，再求和。
2. 实现一行一个 block 的 softmax，复用 block/warp reduction。
3. 针对小行宽与大行宽使用至少两种 launch 策略。
4. 可选实现 mask 或 scale+mask+softmax 融合，比较多 kernel 与融合版本。

**必须掌握知识点**：

- 减 max 防止 exp overflow；全 mask 行等异常语义需定义。
- 行级 max/sum 两次规约和线程内局部累加。
- 行宽决定并行策略；没有万能 block size。
- 融合减少中间写回和 launch，但增加寄存器/代码复杂度。

**可验收标准**：

- 输出有限、每行和接近 1，与参考实现满足 dtype 容差。
- 覆盖行宽 1、31、32、33、128、1000、4096。
- 至少两种策略有形状分界依据。
- 若实现融合版，确认语义与未融合链路完全一致。

**代码产物**：

- `kernels/softmax.cu`
- `tests/test_softmax.cu`
- `benchmarks/bench_softmax.cu`
- `reports/day19_softmax.csv`

#### Day 20｜里程碑 2：常用算子包 v0.2

**工程目标**：把原语和 DL 算子组合成可选型、可验证、可比较的算子包。

**任务**：

1. 统一 GEMM、Conv2D、Softmax 以及 reduction/scan 的接口、shape 检查和 dtype 契约。
2. 为每个算子建立权威参考、参数化测试、朴素/优化/库三路 benchmark。
3. 建立一张“调用库还是自写”的决策表：性能、融合、特殊 shape、维护成本。
4. 随机选两个新 shape，完成从正确性到 profiler 结论的完整演练。

**必须掌握知识点**：

- 算子契约包括 shape、layout、dtype、设备、连续性、别名和误差。
- 成熟库是默认基线，不把“手写 kernel”当目标本身。
- shape-specific 优化必须有 dispatch 与 fallback。
- 报告应同时暴露失败形状和性能差的形状。

**可验收标准（第 2 里程碑）**：

- 四类算子在测试矩阵中全部通过，无 sanitizer 错误。
- benchmark 能一键生成按算子/shape/dtype 汇总的 CSV。
- 每个优化版至少有一个明确的适用区间，区间外自动 fallback。
- 能在 45 分钟内为一个新算子形状建立参考、测量并给出初步瓶颈结论。

**代码产物**：

- `include/operators/`
- `tests/test_operator_suite.cu`
- `benchmarks/bench_operator_suite.cu`
- `reports/milestone_02/operator_matrix.csv`
- `reports/milestone_02/library_vs_custom.md`

---

## 第三大框：PyTorch 接入与可交付工程（第 21～30 天）

### 第三大框通过标准

第 30 天结束时，能够把 CUDA 算子作为 PyTorch 可安装模块交给别人使用：遵循当前 device/stream，支持声明的 dtype/shape，必要时支持 backward，具有自动化正确性测试、benchmark、sanitizer 检查和清楚的 fallback。做到这里，已经具备比较扎实的 CUDA 算子工程作品能力。

### 第 5 周：PyTorch CUDA Extension 最短生产路径

#### Day 21｜LayerNorm/RMSNorm 与 Elementwise Fusion

**工程目标**：完成最适合接入 PyTorch 的融合算子候选。

**任务**：

1. 实现 row-wise LayerNorm 或 RMSNorm 的 CPU/PyTorch 参考。
2. CUDA 版复用 warp/block reduction 计算统计量并完成归一化。
3. 支持可选 affine，尝试融合 residual 或 bias。
4. 对 hidden size 128～8192、不同 batch/row 数做测试和 benchmark。

**必须掌握知识点**：

- LayerNorm 的 mean/variance 与 RMSNorm 的平方均值；`epsilon` 的位置。
- 统计量使用 FP32 累加，输入输出可为低精度。
- 单 kernel 融合的内存流量收益和寄存器压力。
- hidden size/row count 决定一行一个 block 是否合适。

**可验收标准**：

- FP32 及可用的 FP16 结果与 PyTorch 参考匹配。
- 小 batch、大 hidden 和非 2 的幂 hidden 均正确。
- 融合版与未融合链路语义一致，并有 kernel 数/内存流量对比。
- 形成后续 Extension 的主算子候选。

**代码产物**：

- `kernels/norm.cu`
- `tests/test_norm.cu`
- `benchmarks/bench_norm.cu`
- `reports/day21_norm.csv`

#### Day 22｜Extension 骨架与第一个可从 Python 调用的算子

**工程目标**：把已有 elementwise 或 reduction kernel 暴露为 PyTorch 自定义算子。

**任务**：

1. 建立 `pytorch_ext/` 构建与安装骨架，明确开发期 JIT 与发布期 wheel/包构建方式。
2. 注册一个简单 CUDA op，完成 Python → dispatcher/binding → C++ wrapper → CUDA kernel。
3. 检查 tensor 是 CUDA、dtype 支持、shape 合法；先明确要求 contiguous。
4. 写 Python smoke test，并与纯 PyTorch 实现比较。

**必须掌握知识点**：

- PyTorch custom operator/dispatcher 与 CUDA 实现的基本分层。
- `data_ptr<T>()`、shape/stride/dtype/device 信息获取。
- host wrapper 负责验证，kernel 负责计算。
- 不依赖已逐步淘汰的旧式 API 习惯；注册与构建方式保持清晰。

**可验收标准**：

- 可安装后在新 Python 进程 import 并调用。
- CPU tensor、错误 dtype、错误 shape 都得到明确异常。
- CUDA tensor结果与 PyTorch 参考一致。
- Python 测试可独立运行，不依赖手工复制动态库。

**代码产物**：

- `pytorch_ext/setup.*` 或 `pytorch_ext/pyproject.toml`
- `pytorch_ext/csrc/op.cpp`
- `pytorch_ext/csrc/op_cuda.cu`
- `pytorch_ext/tests/test_smoke.py`

#### Day 23｜Current Device、Current Stream 与非连续输入契约

**工程目标**：让 Extension 在真实 PyTorch 程序中不因 stream/device 假设产生隐蔽错误。

**任务**：

1. 获取并使用 PyTorch 当前 CUDA stream，不硬编码默认 stream。
2. 使用 device guard，测试多张 GPU 时的当前 device；单卡则用模拟错误路径验证。
3. 对非连续 tensor 选择：显式 contiguous copy、支持 stride，或清楚拒绝，并测试。
4. 在自建非默认 stream 上构造数据依赖，验证无全局同步且结果正确。

**必须掌握知识点**：

- PyTorch 当前 stream/device 与 CUDA 默认值不能混为一谈。
- tensor lifetime、异步 kernel 和 allocator 的关系。
- contiguous、stride、storage offset 与 view。
- Extension 不应随意 `cudaDeviceSynchronize()`。

**可验收标准**：

- 非默认 stream 测试重复运行无偶发错误。
- 正确处理当前 GPU；无跨设备非法指针访问。
- 非连续输入行为写入接口契约并被测试覆盖。
- Nsight Systems 时间线无无意义的全设备同步。

**代码产物**：

- `pytorch_ext/csrc/tensor_checks.h`
- `pytorch_ext/tests/test_stream_device.py`
- `reports/day23_extension_timeline.nsys-rep`

#### Day 24｜Forward/Backward、Autograd 与混合精度

**工程目标**：为一个训练场景算子提供可验证的梯度链路。

**任务**：

1. 从 fused activation、RMSNorm 或另一可控算子中选择一个，定义 forward/backward 数学契约。
2. 编写 CUDA backward 或通过已注册算子组合完成 backward。
3. 使用 `gradcheck`（双精度小输入或合适策略）及与 PyTorch reference 梯度对比。
4. 声明 autocast/FP16 策略，统计量或累加使用 FP32。

**必须掌握知识点**：

- forward 保存哪些 tensor/统计量会影响显存和 backward 计算。
- autograd 注册、梯度 shape/dtype/device 契约。
- 数值梯度检查的限制与容差。
- mixed precision 下输入 dtype、计算 dtype、输出 dtype 的设计。

**可验收标准**：

- forward 与 reference 匹配，backward 对所有需要梯度的输入匹配。
- gradcheck 通过或对不适用原因有可重复的替代验证。
- `requires_grad=False`、空 tensor、不同 batch shape 行为正确。
- autocast 下无无意的低精度统计量。

**代码产物**：

- `pytorch_ext/csrc/fused_op_cuda.cu`
- `pytorch_ext/fused_op.py`
- `pytorch_ext/tests/test_autograd.py`
- `reports/day24_gradient_contract.md`

#### Day 25｜PyTorch Benchmark 与框架级公平比较

**工程目标**：判断自定义算子在真实调用方式下是否值得存在。

**任务**：

1. 使用 PyTorch benchmark 工具或 CUDA Event 建立 warm-up、同步、重复采样。
2. 比较 eager PyTorch、可用的编译/融合路径、成熟库和自定义 Extension。
3. 覆盖典型 shape、dtype、contiguous/non-contiguous 和 batch 规模。
4. 分别测 kernel-only、算子调用和包含必要 contiguous copy 的真实端到端成本。

**必须掌握知识点**：

- Python 调用开销、小 kernel launch overhead 与大 workload 的差别。
- 公平比较必须保持语义、dtype、layout 和同步范围一致。
- 框架可能自动融合/选择库，reference 不等于低效。
- p50/p95 与离群值比单次最小时间更可靠。

**可验收标准**：

- benchmark 结果包含版本、硬件、shape、dtype、统计方法。
- 至少 12 组 workload，自动标记加速和回退。
- 对“自定义 kernel 更快但端到端更慢”的案例能解释。
- 得出 Extension 值得保留的明确适用区间。

**代码产物**：

- `pytorch_ext/benchmarks/bench_ops.py`
- `pytorch_ext/benchmarks/shapes.json`
- `reports/day25_pytorch_benchmark.csv`
- `reports/day25_findings.md`

### 第 6 周：测试、回归、分派与部署质量

#### Day 26｜测试矩阵与 Compute Sanitizer

**工程目标**：把“我机器上能跑”升级为系统化质量保障。

**任务**：

1. 建立 shape × dtype × layout × device × edge case 的参数化测试矩阵。
2. 加入空 tensor、零维长度、超大索引风险、非法参数和数值极端值。
3. 对小而全的核心 case 跑 memcheck、racecheck、initcheck、synccheck。
4. 设定固定随机种子，并在失败时输出可复现参数。

**必须掌握知识点**：

- 单元测试、性质测试、差分测试和 sanitizer 各自覆盖什么。
- 非确定性、原子规约顺序与浮点容差。
- 32 位索引溢出与 64 位尺寸的风险。
- sanitizer 用例要小，常规 CI 与深度检查可分层。

**可验收标准**：

- 每个公开算子至少覆盖正常、边界、非法、数值四类测试。
- sanitizer 核心集合零错误。
- 失败日志能直接复制 shape/dtype/seed 复现。
- 不支持的输入均显式失败，不出现 silent wrong result。

**代码产物**：

- `pytorch_ext/tests/test_matrix.py`
- `scripts/run_sanitizers.*`
- `reports/day26_test_matrix.md`
- `reports/day26_sanitizer.log`

#### Day 27｜Benchmark 自动化与性能回归门槛

**工程目标**：让未来改代码时自动发现明显退化，而不是凭体感。

**任务**：

1. 定义稳定 workload 集与基线文件，输出 JSON/CSV。
2. 自动计算相对基线变化，对足够长的 kernel 设置合理回归阈值。
3. 固定 GPU 并记录环境；区分噪声、冷启动和真实退化。
4. 故意加入一个慢路径，确认回归检查会失败，随后修复。

**必须掌握知识点**：

- 微基准噪声、频率/温度影响、重复次数和统计阈值。
- 性能门槛应针对稳定 case，不能把每个微小波动当失败。
- correctness regression 与 performance regression 必须分别报告。
- 基线与硬件/软件环境绑定。

**可验收标准**：

- 一条命令生成新结果并与匹配环境的基线比较。
- 故意退化能被检测，恢复后通过。
- 报告列出绝对时间、相对变化和统计波动。
- 没有拿不同 GPU 或不同 dtype 的基线直接比较。

**代码产物**：

- `scripts/run_benchmarks.*`
- `scripts/compare_baseline.*`
- `benchmarks/baselines/`
- `reports/day27_regression_demo.md`

#### Day 28｜模板化、dtype/shape Dispatch 与可靠 Fallback

**工程目标**：把一个演示 kernel 改造成支持有限但清晰的产品接口。

**任务**：

1. 为主算子增加 FP32/FP16（硬件与需求合适时 BF16）dispatch。
2. 按 hidden size/行宽/对齐条件选择 2～3 个 kernel specialization。
3. 对其余 shape 使用通用 CUDA kernel 或 PyTorch/reference fallback。
4. 记录每条 dispatch 条件，并做边界两侧测试和 benchmark。

**必须掌握知识点**：

- 编译期模板与运行时 dispatch 的分工。
- specialization 必须与 shape 数据分布匹配。
- vectorized path 的对齐、长度和别名条件。
- fallback 是可靠性设计，不是失败。

**可验收标准**：

- 每条 dispatch 分支都被测试命中，边界不存在空洞。
- 不满足 fast-path 前提时不会错误进入向量化 kernel。
- fallback 结果正确，并在性能报告中标出。
- 二进制/编译时间未因无节制模板组合显著膨胀。

**代码产物**：

- `pytorch_ext/csrc/dispatch.h`
- `pytorch_ext/csrc/op_kernels.cuh`
- `pytorch_ext/tests/test_dispatch.py`
- `reports/day28_dispatch_map.md`

#### Day 29｜内存分配、CUDA Graph 与 Launch 开销

**工程目标**：优化小算子流水线中的分配与 launch 开销，并明确适用限制。

**任务**：

1. 用 Nsight Systems 找出循环内 `cudaMalloc/cudaFree`、同步和密集小 kernel。
2. 复用 workspace，或在合适环境试用 stream-ordered allocation。
3. 对 shape/地址稳定的重复执行链尝试 CUDA Graph capture/replay。
4. 比较 eager launch 与 graph replay；测试 shape 改变时的 fallback。

**必须掌握知识点**：

- 分配/释放可能带来同步；workspace 生命周期与并发安全。
- CUDA Graph 适合拓扑稳定、重复执行、launch-bound 的链路。
- capture 限制、stream 语义和输入地址稳定性。
- graph 对大而计算密集 kernel 的收益通常有限。

**可验收标准**：

- profiler 时间线中明确指出至少一处 host/launch 开销。
- 循环内无不必要的分配。
- graph 版结果一致，重复执行有公平数据。
- 不满足 capture 条件时能可靠回退，不把 graph 强套到所有场景。

**代码产物**：

- `apps/cuda_graph_pipeline.cu` 或 `pytorch_ext/graph_demo.py`
- `benchmarks/bench_launch_overhead.*`
- `reports/day29_graph_timeline.nsys-rep`
- `reports/day29_graph_findings.md`

#### Day 30｜里程碑 3：可安装 CUDA Extension v0.3

**工程目标**：交付一个可安装、可测试、可 benchmark、可定位问题的 PyTorch CUDA 模块。

**任务**：

1. 选定一个主算子（推荐 RMSNorm/LayerNorm、Softmax 或融合 elementwise），整理 forward、可选 backward 和 dispatch。
2. 从干净环境执行 build/install/import/test，记录兼容的 PyTorch/CUDA/GPU 范围。
3. 跑完整正确性矩阵、核心 sanitizer、性能回归和 Nsight 分析。
4. 写使用示例、支持范围、fallback、已知限制和复现实验命令。

**必须掌握知识点**：

- 一个可交付算子同时需要 API 契约、正确性、性能、兼容性和诊断材料。
- 当前 stream/device、dtype、layout 与 autograd 是框架集成底线。
- benchmark 只承诺已测环境和 shape，不做泛化夸张。
- 版本信息和失败信息必须对使用者可见。

**可验收标准（第 3 里程碑）**：

- 新环境按说明可安装，示例一次运行成功。
- 测试矩阵全部通过，sanitizer 零错误，回归检查通过。
- 至少一个真实适用区间达到稳定收益；其他区间有正确 fallback。
- 报告可回答：为何自写、何时更快、何时不用、如何定位错误。

**代码产物**：

- `pytorch_ext/dist/` 或可重复构建配置
- `pytorch_ext/examples/basic_usage.py`
- `reports/milestone_03/test_report.md`
- `reports/milestone_03/benchmark.csv`
- `reports/milestone_03/profile.*-rep`

---

## 第四大框：高级补强与完整作品（第 31～40 天）

### 第四大框通过标准

第 40 天结束时，不要求手写出超过所有成熟库的 kernel，而要能够对一个真实算子做完整工程决策：建立基线、分类瓶颈、选择优化手段、管理架构差异、接入框架、证明正确性和收益，并把结果交给他人复现。

### 第 7 周：性能模型、硬件特性与扩展知识

#### Day 31｜Roofline 思维：先给瓶颈分类

**工程目标**：在修改 kernel 前，用算术强度和实测吞吐判断优化方向。

**任务**：

1. 选择 vector op、reduction、GEMM 三个代表 kernel，估算 FLOPs 和最少字节流量。
2. 计算 arithmetic intensity、实测带宽和实测 FLOP/s。
3. 结合硬件理论/实测上限画简化 roofline 或表格。
4. 为每个 kernel 给出优先优化项，并用一次小实验验证判断。

**必须掌握知识点**：

- arithmetic intensity = FLOPs/byte；内存上限与计算上限。
- 理论峰值只是上界，实测可达带宽更适合作参照。
- cache/重复读取会让字节估算需要声明假设。
- roofline 用于方向判断，不是精确预测每个周期。

**可验收标准**：

- 三个 kernel 的 FLOP/byte 估算过程可检查。
- 性能单位、数据量和时间换算无误。
- 每个 kernel 被归为 bandwidth/compute/latency 或混合瓶颈并有证据。
- 优化实验与预测方向一致，或能解释偏差。

**代码产物**：

- `scripts/roofline_report.*`
- `reports/day31_roofline.csv`
- `reports/day31_roofline.md`

#### Day 32｜Occupancy、寄存器与 Shared Memory 调参

**工程目标**：理解资源约束，避免把 occupancy 当唯一目标。

**任务**：

1. 对 tiled GEMM、softmax/norm 采集寄存器数、shared memory、active warps 和 spills。
2. 改变 block/tile/thread-coarsening 配置，得到资源与性能矩阵。
3. 使用 launch bounds 或寄存器限制做小实验，观察 spill 与性能。
4. 选出稳定配置，并记录它为何适合目标 shape。

**必须掌握知识点**：

- SM 上线程、block、warp、寄存器和 shared memory 的共同约束。
- theoretical/achieved occupancy、ILP、register spill。
- 限制寄存器可能提升 occupancy，也可能因 local memory 变慢。
- 配置选择必须结合 shape 与 stall 指标。

**可验收标准**：

- 至少 6 组配置有资源和性能数据。
- 能指出限制 active blocks 的主要资源。
- 不以最高 occupancy 自动作为最快配置。
- 最终默认配置与 fallback 条件写入 dispatch map。

**代码产物**：

- `benchmarks/bench_resource_tuning.cu`
- `reports/day32_resource_matrix.csv`
- `reports/day32_profile.ncu-rep`

#### Day 33｜Tensor Core：会使用，不陷入手写汇编

**工程目标**：理解 Tensor Core 的工程入口和精度约束，优先通过库/模板库使用。

**任务**：

1. 用 cuBLASLt 或 CUTLASS 跑 FP16/BF16 输入、FP32 累加 GEMM 基线。
2. 检查 shape、alignment、layout 对 Tensor Core 路径的要求。
3. 可选写一个最小 WMMA 示例，重点验证 fragment/layout/边缘限制。
4. 与 FP32 CUDA Core 路径比较性能、误差和转换成本。

**必须掌握知识点**：

- Tensor Core 依赖 GPU 架构、dtype、shape/layout 和库算法选择。
- 输入精度与累加精度的区别；TF32/FP16/BF16 语义不能混称。
- 库/模板库通常比手写 WMMA/PTX 更适合工程。
- padding/转换成本可能抵消小矩阵收益。

**可验收标准**：

- 至少一组满足条件的 workload 确认走到 Tensor Core 路径。
- 与 FP32 reference 的误差和性能均有数据。
- 不支持硬件能跳过测试并清楚说明，而不是构建失败。
- 能给出生产中选择 cuBLASLt/CUTLASS/自写的理由。

**代码产物**：

- `apps/tensor_core_gemm.cu`
- `benchmarks/bench_tensor_core.cu`
- `reports/day33_tensor_core.csv`
- `reports/day33_selection.md`

#### Day 34｜异步拷贝与软件流水（架构条件化）

**工程目标**：理解 global→shared 数据流水，学会为新架构优化同时保留 fallback。

**任务**：

1. 在 tiled GEMM 或 stencil 类 kernel 中找 load/compute 交替位置。
2. 对支持的架构试用 `memcpy_async`/pipeline 或相应模板库能力；不支持则分析 CUTLASS 示例和 profiler。
3. 实现双缓冲概念版本，比较同步 tile load 与流水版。
4. 添加编译期架构保护和通用 fallback。

**必须掌握知识点**：

- load/compute overlap、双缓冲、pipeline stage。
- 异步拷贝有对齐、粒度、架构与同步要求。
- 增加 stage 会消耗 shared memory/寄存器，不一定更快。
- 架构专用 fast path 必须与可移植 fallback 共存。

**可验收标准**：

- 支持硬件上有 before/after profiler；不支持时完成可编译 fallback 和分析报告。
- 结果对边缘 tile 仍正确。
- 架构判断不会让其他 GPU 加载非法 binary/path。
- 能说明收益或无收益与资源/延迟隐藏的关系。

**代码产物**：

- `kernels/pipelined_tile.cu`
- `tests/test_pipelined_tile.cu`
- `reports/day34_pipeline.ncu-rep`
- `reports/day34_arch_fallback.md`

#### Day 35｜多 GPU/NCCL 入门与优先级边界

**工程目标**：补齐多 GPU 工程常识，但不把它抢到单卡核心能力之前。

**任务**：

1. 枚举设备，理解每进程/线程 current device 和跨设备指针错误。
2. 两卡可用时做 peer access/peer copy 小实验；单卡则完成可跳过测试。
3. 两卡与 NCCL 可用时完成 all-reduce 示例并验证；否则阅读接口并实现环境检测。
4. 用 Nsight Systems 查看通信与计算时间线，尝试一次 chunk overlap。

**必须掌握知识点**：

- device context、P2P 能力、PCIe/NVLink 拓扑。
- NCCL collective 的 rank、communicator、stream 语义。
- 通信/计算重叠依赖分块与数据依赖。
- 单机多卡只是起点；分布式容错/调度不在本 40 天核心范围。

**可验收标准**：

- 单卡环境不失败，清楚输出“跳过及原因”。
- 多卡环境下结果正确、资源正确释放、无错误 device 使用。
- 能从时间线区分 H2D、P2P/NCCL 和 compute。
- 写明何时值得继续深入多 GPU。

**代码产物**：

- `apps/multi_gpu_probe.cu`
- `apps/nccl_allreduce_demo.cu`（环境可用时）
- `reports/day35_multi_gpu.md`

### 第 8 周：完整 Capstone 作品

> 推荐主项目：**PyTorch Fused RMSNorm/LayerNorm Extension**。它同时覆盖规约、低精度、融合、stream、dispatch、autograd、测试和 benchmark，工作量比从零打造高性能 GEMM/Attention 更可控。若业务更偏图像，也可替换为 fused bias + activation 或特定 shape Conv2D，但验收结构保持一致。

#### Day 36｜定义真实需求、边界与基线

**工程目标**：把“做一个快 kernel”变成可验证的产品问题。

**任务**：

1. 冻结算子语义：公式、输入输出、shape、dtype、layout、device、forward/backward、异常行为。
2. 收集 12～20 个真实或代表性 shape，标记高频与极端 case。
3. 建立 PyTorch/成熟库正确性参考和性能基线。
4. 写出成功标准：正确性容差、支持范围、目标 shape 收益、fallback 和交付物。

**必须掌握知识点**：

- 需求先于优化；没有 workload 分布就没有合理 specialization。
- 语义一致是性能比较前提。
- 目标可以是 latency、throughput、显存或减少 launch，不只“最快”。
- 明确不做的范围能防止项目失控。

**可验收标准**：

- 规格表无未定义输入，至少包含 12 个 workload。
- 每个 workload 都有 reference 输出和基线时间。
- 成功标准可量化，且不承诺所有 shape 都超过框架。
- 明确列出 v1 不支持的功能与 fallback。

**代码产物**：

- `capstone/SPEC.md`
- `capstone/workloads.json`
- `capstone/reference.py`
- `reports/capstone/baseline.csv`

#### Day 37｜正确的通用 Kernel v1

**工程目标**：先覆盖全部声明输入，建立稳固 fallback，再考虑 fast path。

**任务**：

1. 实现通用 CUDA forward，复用已验证的规约和错误检查组件。
2. 若项目要求训练，实现 backward；否则明确 inference-only。
3. 建立参数化差分测试与数值压力测试。
4. 跑 sanitizer，并修复所有越界、竞争、同步和未初始化问题。

**必须掌握知识点**：

- 通用 kernel 的职责是可靠覆盖，不必是最终最快版本。
- 复用经过验证的 primitive，避免重新复制旧 bug。
- 误差阈值与 accumulation dtype 是接口一部分。
- backward 的性能与正确性需要单独测。

**可验收标准**：

- 规格内 workload 全部正确，规格外输入按约定 fallback/报错。
- sanitizer 全部通过。
- forward 以及声明支持的 backward 均有差分测试。
- 形成不会被后续 fast path 破坏的可信通用基线。

**代码产物**：

- `capstone/csrc/kernel_generic.cu`
- `capstone/tests/test_correctness.py`
- `capstone/tests/test_edge_cases.py`
- `reports/capstone/day37_sanitizer.log`

#### Day 38｜Profile 驱动的 Fast Path

**工程目标**：只针对真实高频 shape 做 1～2 个有证据的优化版本。

**任务**：

1. 用 Nsight Systems 判断端到端是否 launch-bound，用 Nsight Compute 判断 kernel 瓶颈。
2. 从合并访存、warp reduction、vectorized load、融合、tile/occupancy 中选择最匹配的优化。
3. 实现 1～2 个 specialization，并添加严格 dispatch 前提。
4. 跑全 workload before/after，检查目标外 shape 无退化或正确 fallback。

**必须掌握知识点**：

- 每个优化必须对应 profiler 证据。
- fast path 条件包含 shape、dtype、alignment、layout 和架构。
- 局部加速不能以破坏尾部、精度或其他 shape 为代价。
- 一次只改变主要变量，便于归因。

**可验收标准**：

- 报告完整呈现“指标 → 假设 → 修改 → 数据”。
- 高频目标 shape 有稳定收益或得到可信的停止优化结论。
- 全量 correctness 继续通过；非目标 shape 走通用实现。
- 无只展示最佳单点、隐藏退化 shape 的选择性汇报。

**代码产物**：

- `capstone/csrc/kernel_fast.cu`
- `capstone/csrc/dispatch.cpp`
- `reports/capstone/day38_before_after.csv`
- `reports/capstone/day38_analysis.md`

#### Day 39｜框架集成、自动化与交付演练

**工程目标**：在干净环境模拟他人使用并补齐所有工程缺口。

**任务**：

1. 完成 PyTorch 注册、current stream/device、autograd/AMP 和 fallback 集成。
2. 建立一键安装、正确性测试、sanitizer 子集、benchmark、基线对比。
3. 在新的虚拟环境或干净构建目录完成安装和示例运行。
4. 邀请“未来的自己”按 README 操作，只依据错误日志定位一个故意注入的问题。

**必须掌握知识点**：

- 交付不是复制 `.so/.dll`，而是可复现构建与明确兼容范围。
- 自动化脚本退出码、错误日志和环境信息是可维护性的核心。
- 性能基线应与硬件环境绑定。
- fallback 与报错信息决定真实可用性。

**可验收标准**：

- 干净环境一次安装成功，示例和测试通过。
- 一条命令生成 correctness + benchmark 汇总。
- 故意错误可由日志快速定位，恢复后全绿。
- README 中所有命令均实际执行验证过。

**代码产物**：

- `capstone/pyproject.toml` 或等价构建配置
- `capstone/scripts/verify_all.*`
- `capstone/examples/`
- `reports/capstone/day39_release_checklist.md`

#### Day 40｜里程碑 4：发布 v1.0 与能力复盘

**工程目标**：产出一个可展示、可复现、不过度宣传的完整 CUDA 工程作品。

**任务**：

1. 冻结代码，运行全量测试、sanitizer、benchmark 和 profiler，保存环境快照。
2. 汇总 correctness、p50/p95 latency、端到端性能、显存、适用/不适用 shape。
3. 写技术报告：需求、设计、关键错误、profiling 证据、优化迭代、局限和下一步。
4. 做一次 60 分钟限时演练：增加新 shape，判断走哪个 kernel，验证并给出性能结论。

**必须掌握知识点**：

- 工程结论必须可复现、可证伪、限定适用范围。
- “没有超过成熟库”也可以是成功结论，只要定位和决策可靠。
- 最有价值的能力是快速完成正确性—测量—定位—优化闭环。
- 后续学习由项目暴露的问题驱动，不再平均用力。

**可验收标准（最终里程碑）**：

- 在声明环境中可构建、安装、运行；所有必需测试和 sanitizer 通过。
- 目标 workload 有完整基线与 v1.0 数据，报告不隐藏退化项。
- 每个 fast path 都有 dispatch 条件和通用 fallback。
- 另一位开发者只看说明即可复现实验。
- 限时演练能在 60 分钟内完成新 shape 的正确性与初步性能判断。

**代码产物**：

- `capstone/README.md`
- `capstone/CHANGELOG.md`
- `capstone/VERSION`
- `reports/milestone_04/final_report.md`
- `reports/milestone_04/final_benchmark.csv`
- `reports/milestone_04/final_profile.*-rep`

---

## 四个里程碑总表

| 时间点 | 能力结果 | 必须拿得出的证据 | 此时可以做什么 |
| --- | --- | --- | --- |
| Day 10 | CUDA 核心闭环 | 3 类 kernel、测试、benchmark、sanitizer、Nsight 报告 | 独立完成简单 elementwise/transpose/reduction 类需求 |
| Day 20 | 原语与常用 DL 算子 | scan/histogram、GEMM/Conv/Softmax、库对比与选型表 | 做算子原型，判断库调用、自写、融合或 fallback |
| Day 30 | 可交付 PyTorch Extension | 可安装包、stream/device、dtype dispatch、autograd、测试与性能回归 | 交付一个真实框架自定义算子，形成有分量的项目作品 |
| Day 40 | 完整工程作品 | 规格、通用版、fast path、profiling 证据、自动化、最终报告 | 独立承担 CUDA 算子的实现、诊断、优化、集成与交付 |

## 优先级：哪些必须先掌握，哪些以后慢慢补

### 第一优先级（Day 1～10，绝不跳过）

- 索引、边界、grid-stride loop。
- 显存生命周期、H2D/D2H、错误检查。
- 合并访存、shared memory、规约、warp shuffle。
- 正确性参考、浮点容差、CUDA Event benchmark。
- Compute Sanitizer、Nsight Compute、Nsight Systems。
- stream、pinned memory、异步拷贝与同步关系。

### 第二优先级（Day 11～30，工程竞争力）

- atomic、scan、CUB/Thrust，库优先的选型能力。
- layout/stride、FP16/BF16、GEMM、Conv、Softmax、Norm/Fusion。
- PyTorch custom operator、current stream/device、autograd/AMP。
- 参数化测试、sanitizer、性能基线、dispatch 与 fallback。

### 第三优先级（Day 31～40 或项目需要时）

- Roofline、资源调参、Tensor Core、异步 pipeline。
- CUDA Graph、异步 allocator、多 GPU/NCCL。
- CUTLASS/cuBLASLt 更深用法。

### 暂时不建议占用主线时间

- 大量手写 PTX/SASS。
- 一开始就挑战生产级 FlashAttention 或库级 GEMM。
- 背诵所有 GPU 微架构参数、所有 Nsight 指标。
- 在没有真实 workload 前做过度模板元编程和海量 shape specialization。
- 为“手写而手写”，忽略 cuBLAS、cuDNN、CUB、Thrust、CUTLASS 等成熟方案。

## 40 天结束后的按需补充顺序

1. **若目标是深度学习算子岗**：CUTLASS/cuBLASLt → attention/normalization/fusion → Triton 对照 → 更完整 AMP/autograd。
2. **若目标是 HPC**：多维 stencil → 稀疏矩阵/cuSPARSE → MPI+NCCL → 多 GPU 域分解。
3. **若目标是推理部署**：TensorRT plugin → CUDA Graph → memory pool → dynamic shape 与量化。
4. **若目标是性能专家**：更深入 Nsight 指标 → SASS 阅读 → 架构特定流水 → 必要时再学 PTX。

## 每周五分钟自检

- 我本周新增的每个 kernel 是否都有参考实现？
- 是否测了非 block 整数倍、极小和极大输入？
- “更快”是否来自公平、稳定、可复现的 benchmark？
- 我能否用一个具体 profiler 指标说明瓶颈？
- 是否优先评估了成熟库，而不是默认重写？
- 新 fast path 是否有严格条件和可靠 fallback？
- 本周代码是否能由一条命令构建、测试并复现实验？

---

## 附录 A：Capstone 项目验收规范

### 1. 功能与正确性

- 至少包含三种模式中的两种：elementwise/fusion、reduction、stencil/tiling。
- CPU/reference 与 CUDA 输出自动比对。
- 覆盖常规、边界、非对齐、非 tile 倍数、大输入。
- 明确 dtype 和误差准则。
- 固定随机种子。
- sanitizer 无未解释错误。
- 发生 CUDA 错误时输出 API、文件、行号和可读错误信息，并返回失败状态。

### 2. 性能版本

至少保留：

1. CPU 或已知正确 reference；
2. naive CUDA；
3. optimized CUDA；
4. 能找到成熟库时，再加 library baseline。

不得只保留最终最快代码，否则无法说明优化过程。

### 3. 工程质量

- CMake 一键配置和构建。
- README 包含依赖、硬件、构建、运行、测试、benchmark、已知限制。
- kernel、host orchestration、reference、tests、benchmark 分离。
- device memory、streams、events、library handles 使用 RAII 或等价可靠生命周期管理。
- 支持命令行设置数据规模、迭代次数、实现版本；不改源码即可复现实验。
- Release 和 Debug/Sanitizer 流程分开。
- 至少一个“干净目录从零构建”的验收记录。

### 4. Profiling 证据

- 一份 Nsight Systems 报告：能看到 H2D、kernel、D2H、同步与并发。
- 关键 kernel 的 Nsight Compute before/after。
- 文档中列出：
  - 原瓶颈；
  - 指标证据；
  - 修改；
  - 时间变化；
  - 相关指标变化；
  - 是否引入正确性、精度或可维护性代价。

### 5. 求职表达

必须能在 5 分钟内回答：

1. 项目解决什么问题？
2. 为什么适合 GPU？
3. CPU/reference 和 naive CUDA 分别是什么？
4. profiler 发现的真正瓶颈是什么？
5. 做了什么优化，为什么有效？
6. GPU-only 和端到端各快多少？
7. 如何验证结果正确？
8. 哪次优化失败了，为什么？
9. 换一张 GPU，哪些结论可能变化？
10. 生产环境中会直接用库还是保留自写 kernel？

---

## 附录 B：统一 Benchmark 规范

### A. 实验环境必须记录

- GPU 完整型号与显存
- driver 版本
- CUDA Toolkit / `nvcc` 版本
- 操作系统和 host compiler
- 编译类型与关键编译选项
- CUDA architecture
- dtype
- 输入 shape/规模
- ECC、功耗/频率限制等能取得的信息

### B. 构建和运行条件

- 使用 Release 优化构建；Debug 数字不得用于性能结论。
- benchmark 时减少后台 GPU 负载。
- 初始化 CUDA context、库 handle、JIT/cold start 开销应单独报告或在正式热态 benchmark 前完成。
- 分配和初始化原则上移出重复计时循环；若项目关心 allocator，则另设 allocator benchmark。
- profiler 会引入开销，profile run 与正式 benchmark run 分开。

### C. 正确性先行

- 每种实现、每种 dtype、每个输入规模先过 correctness，再加入统计。
- 失败样本不得从结果中静默删除。
- 浮点误差阈值写入报告；更换精度必须重新验证。

### D. 计时方法

- GPU kernel / 同 stream 工作段：使用 CUDA Events。
- 端到端：使用 host wall-clock，并在测量终点正确同步。
- 明确以下项目是否计入：
  - host allocation
  - pinned registration/allocation
  - device allocation
  - H2D
  - kernel
  - D2H
  - 同步
  - 数据预/后处理
- 至少同时提供：
  1. kernel-only 或 GPU pipeline 时间；
  2. end-to-end 时间。

### E. 预热与统计

- 预热建议 10～50 次，直到初始化和频率爬升影响明显减小。
- 正式重复至少 30 次，或让累计计时时间足够长（建议不少于约 1 秒）。
- 报告 median；同时报告 p10/p90、p5/p95 或标准差中的一种。
- 不用“最快的一次”代表常规性能。
- 极短 kernel 应批量重复后除以次数，避免 event 分辨率和 launch 抖动支配结果。

### F. 输入规模

- 至少 small / medium / large 三档。
- 至少一个不规则尺寸，不允许全部是 2 的幂或 tile 的整数倍。
- 端到端比较必须处理同样的数据量、执行同样的数学操作。
- 只在大规模上快，要如实展示交叉点，不能隐去小规模结果。

### G. 基线公平性

- CPU baseline 使用合理 Release 优化，不能拿 Debug 单线程低效代码制造巨大 speedup。
- GPU 与 CPU 使用相同 dtype、相同误差要求和相同计算定义。
- 有成熟库时，提供 CUB/cuBLAS/Thrust 等库基线。
- speedup 写成 `baseline_time / candidate_time`，并明确 baseline 是 CPU、naive CUDA 还是库。
- 自写 GEMM 不应只和朴素三重循环 CPU 比；生产价值必须与 cuBLAS 对照。

### H. 性能指标

- 时间与吞吐量必须至少选一个主指标。
- memory-bound kernel 报有效带宽，并与本机实测 copy 带宽比较。
- compute-bound kernel 报清楚 FLOP 计数约定，必要时给 GFLOP/s。
- latency-sensitive 项目报告 p50/p95；吞吐型项目报告 items/s、GB/s 等。
- 所有百分比和 speedup 都附原始绝对时间，避免“提升 200%”的歧义。

### I. 可复现输出

建议 benchmark 每行输出：

```text
timestamp,gpu,driver,toolkit,git_commit,impl,dtype,shape,warmup,repeats,
kernel_ms_median,kernel_ms_p95,e2e_ms_median,throughput,error_metric
```

原始 CSV/JSON 保留，README 中的表格和图必须能追溯到原始数据和代码版本。

---

## 附录 C：最常见误区与审查红线

1. **把“看完课程”当完成。** 过关依据只能是代码、测试、sanitizer、benchmark、profile 和解释。
2. **只会 kernel 语法，不会工程工具。** 没有构建、错误检查、测试、调试和复现，不算可求职项目。
3. **优化前不做 CPU/reference。** 性能更快但结果错误，没有价值。
4. **在 kernel launch 外包 CPU 计时器却不同步。** 测到的主要是 launch 开销。
5. **只报 kernel speedup。** 应同时报告端到端，否则可能被传输和同步完全抵消。
6. **拿低效 CPU Debug 代码做基线。** 这种几十倍/几百倍 speedup 经不起面试追问。
7. **100% occupancy 崇拜。** occupancy 是延迟隐藏手段，不是最终目标；更高 occupancy 可能伴随更多 spill 或更少每线程资源。
8. **shared memory 崇拜。** 没有复用或重排需求时，额外拷贝与同步可能更慢。
9. **线程越多、block 越大越快。** block size 受 registers、shared memory、同步和工作规模共同影响。
10. **`cudaMemcpyAsync` 一定重叠。** pageable memory、默认 stream、隐式依赖、工作太小或硬件限制都会使其不重叠。
11. **用 `cudaDeviceSynchronize()` 修所有问题。** 它会隐藏依赖设计错误并摧毁并发；应使用 stream/event 表达精确依赖。
12. **Unified Memory 等于零成本统一内存。** page migration 和访问模式仍会影响性能。
13. **atomic 一定不能用。** 低冲突或硬件支持良好时 atomic 很实用；应该测 contention，而不是先入为主。
14. **自写 kernel 必须打败成熟库。** 工程能力包含知道何时调用 CUB/cuBLAS；手写实现主要用于定制、融合或学习。
15. **只测规则尺寸。** `N=1024` 全过不能证明 `N=1025` 正确。
16. **浮点必须 bitwise 相等。** 并行归约顺序会改变舍入；应建立合理误差模型。
17. **反过来把 epsilon 放得很宽。** 宽容差可能掩盖漏算、越界或错误索引。
18. **在分支中错误调用 `__syncthreads()`。** 这是典型同步错误，必须用 synccheck 和边界测试暴露。
19. **照抄旧教程的隐式 warp-synchronous/`volatile` 技巧。** 现代代码要清楚 active mask 与显式 warp 同步。
20. **盲目追某个 profiler 指标。** 所有指标必须回到最终耗时、吞吐和正确性。
21. **一次改五件事。** 无法知道收益来自哪里，也无法在不同 GPU 上迁移结论。
22. **只展示成功优化。** 一份能解释失败实验的报告更像真实工程。
23. **hardcode 自己显卡和固定 shape。** 求职项目应至少能检测设备能力、接受不规则输入并配置 architecture。
24. **异步 buffer 提前释放/复用。** host 函数返回不代表 GPU 已经完成；生命周期必须覆盖异步操作。
25. **两个月同时铺开 CUDA、PyTorch、TensorRT、NCCL、CUTLASS、编译器。** 结果通常是每项只会跑 demo，应坚持一条专项。

---

## 附录 D：最终“是否过关”总检查

满足以下全部条件，才可把路线标记为完成：

- [ ] 能从零建立 CMake CUDA 工程，并在 README 中复现。
- [ ] 能独立写 map、reduction、stencil/tiling 中至少三类 kernel。
- [ ] 任意非整块尺寸、非 2 的幂和边界输入正确。
- [ ] 有 CPU/reference、自动测试和明确浮点误差标准。
- [ ] Compute Sanitizer 对适用测试无未解释错误。
- [ ] 会用 CUDA Event 测 kernel、wall-clock 测 end-to-end。
- [ ] 会计算有效带宽，知道 kernel-only 与 end-to-end 的区别。
- [ ] 会用 Nsight Systems 找系统瓶颈，用 Nsight Compute 解释关键 kernel。
- [ ] 有一次完整 before/after 优化证据和一次失败实验复盘。
- [ ] 能用 pinned memory、streams、events 实现并证明一次流水重叠。
- [ ] 会使用 CUB，并正确调用一次 cuBLAS。
- [ ] capstone 有 reference、naive、optimized、library（适用时）版本。
- [ ] benchmark 遵循预热、重复、统计、环境记录和公平基线规范。
- [ ] 项目支持配置、错误处理和可靠资源释放。
- [ ] 能做 5 分钟项目说明并回答“为什么快、怎么证实、何时不该自写”。

---

## 附录 E：官方资料索引（按需查阅）

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)：当前官方编程模型、异步执行、内存、编译与高级特性的主参考。
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：APOD、正确性、计时、有效带宽、合并访问、occupancy 等工程实践。
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)：memcheck、racecheck、initcheck、synccheck。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：CUDA/NVTX 时间线与系统级分析。
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：kernel 指标、section sets、roofline 和分析方法。
- [CUDA Libraries Documentation](https://docs.nvidia.com/cuda-libraries/)：cuBLAS、cuFFT、cuSPARSE 等官方库入口。

本路线应把这些文档当作“遇到问题时查证的工程手册”。先实现、测试、测量，再按 profiler 指向阅读对应章节，效率远高于连续数周通读理论。
