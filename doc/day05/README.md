# Day 5：全局内存访问、合并访问与 `float4`

> 本日目标：理解“相同的数学运算，为什么仅仅改变线程访问地址就可能快几倍”。本章从 warp、地址分布和内存事务讲起，再逐步分析 contiguous、strided 和 `float4` 三种 AXPY 实现。

## 1. 本章的数学任务：AXPY

AXPY 是一个常见的逐元素运算：

```text
y[i] = alpha × x[i] + y[i]
```

其中：

- `x` 是只读输入数组；
- `y` 既是输入也是输出；
- `alpha` 是一个标量；
- 每个元素彼此独立。

三个 kernel 必须覆盖相同下标集合并产生相同结果。只有在正确性、工作量和计时范围相同的前提下，性能差异才能归因于访问方式。

## 2. 什么是全局内存

Global memory（全局内存）通常指 GPU 的大容量设备内存。`cudaMalloc` 得到的地址位于这里。

特点：

- 容量大，所有线程都可以通过指针访问；
- 相比寄存器和 shared memory，访问延迟高；
- 性能不仅取决于“访问多少数据”，还取决于同一个 warp 的地址能否被硬件高效合并。

全局内存不是 C++ 的“全局变量”概念；这里的“global”描述 CUDA 内存空间。

## 3. 什么是 warp

CUDA 会把一个 block 中相邻的线程按 32 个一组调度，这个执行组叫 warp。

warp 内线程编号常称为 lane：

```text
lane 0, lane 1, lane 2, ... lane 31
```

lane 可由线性线程编号对 32 取余得到：

```cpp
const int lane = linear_index % 32;
```

初学阶段可以把 warp 理解为“硬件一次共同推进的一组 32 个线程”。每个 lane 仍有自己的寄存器和数据地址，但它们通常执行同一条指令。

## 4. 什么是合并访问

当 warp 执行一条全局内存 load/store 指令时，32 个 lane 会同时提出地址请求。硬件会把这些地址覆盖到若干内存事务中。

如果相邻 lane 访问相邻、适当对齐的地址：

```text
lane 0 → x[0]
lane 1 → x[1]
lane 2 → x[2]
...
lane 31 → x[31]
```

这些地址集中，通常可以用较少事务服务，称为**合并访问（coalesced access）**。

如果地址相隔很远：

```text
lane 0 → x[0]
lane 1 → x[1024]
lane 2 → x[2048]
...
```

同一条指令可能需要覆盖很多分散的内存区域，产生更多事务和未使用的数据传输，吞吐通常下降。

### 4.1 不要把合并访问理解成“一次只发一个事务”

实际事务数量与 GPU 架构、访问宽度、地址对齐、缓存层级等有关。初学时应抓住稳定结论：

> warp 内地址越连续且对齐，硬件越容易高效服务；地址越分散，越可能浪费内存事务。

本章 benchmark 能证明性能差异，但没有硬件计数器时，不把具体事务数量写成已验证事实。

## 5. Contiguous：相邻线程访问相邻元素

核心索引：

```cpp
const int index = blockIdx.x * blockDim.x + threadIdx.x;

if (index < n) {
    y[index] = alpha * x[index] + y[index];
}
```

第一个 warp 通常访问：

```text
x[0..31] 和 y[0..31]
```

第二个 warp 访问：

```text
x[32..63] 和 y[32..63]
```

地址连续，因此这是本章的正常基线。

## 6. Strided：故意制造 warp 内分散地址

项目不是简单写 `index * stride`，因为那可能遗漏元素或越界。它把可整除 32 的主体数据看成一个逻辑矩阵，然后改变线性线程编号到数据下标的映射。

核心代码：

```cpp
const int bulk_size = n - n % 32;
const int rows = bulk_size / 32;
const int lane = linear_index % 32;
const int row = linear_index / 32;
const int index = lane * rows + row;
```

### 6.1 每个变量是什么

- `bulk_size`：能被 32 整除的主体元素数；
- `rows`：把主体理解为 `rows × 32` 时的行数；
- `lane`：当前线性线程在 warp 中的位置 0～31；
- `row`：当前线性线程属于第几个 32 元素组；
- `index`：重新排列后真正访问的数据下标。

### 6.2 手算小例子

为了便于手算，假设 warp 宽度是 4（真实 CUDA warp 仍是 32），`bulk_size = 16`：

```text
rows = 16 / 4 = 4
index = lane * 4 + row
```

第 0 组线程的 `row = 0`：

| lane | index |
|---:|---:|
| 0 | 0 |
| 1 | 4 |
| 2 | 8 |
| 3 | 12 |

相邻 lane 的地址相隔 4 个元素。第 1 组访问 `1、5、9、13`。把所有组联合起来仍然覆盖 0～15，每个元素一次。

真实 warp 宽度为 32。当 `n = 1,000,000` 量级时，`rows` 很大，相邻 lane 的地址相隔 `rows` 个 `float`，故意破坏连续性。

### 6.3 尾部为什么仍然正确

`n` 不一定能被 32 整除。主体 `0..bulk_size-1` 使用重映射；`linear_index >= bulk_size` 时保留原线性下标，处理最后 `n % 32` 个元素。

因此 strided 版本改变访问顺序，不改变被处理的元素集合。

## 7. 从“线性索引”到“概念矩阵”

Strided 映射很容易难懂，因为它把同一个线性编号拆成：

```text
row  = linear_index / 32
lane = linear_index % 32
```

这与 Day2 的二维反向映射相同：除法得到第几行，取余得到行内位置。

然后它用：

```text
index = lane × rows + row
```

相当于按另一方向读取这个逻辑矩阵。这里没有真正创建二维数组；“矩阵”只是帮助理解下标置换的模型。

## 8. `float4` 是什么

CUDA 的 `float4` 是包含四个 `float` 分量的向量类型：

```cpp
float4 value;
value.x;
value.y;
value.z;
value.w;
```

一个 `float4` 通常占 16 bytes。将 `float*` 重新解释为 `float4*` 后，一次向量 load/store 可以表达连续四个 `float` 的访问。

```cpp
const float4* x4 = reinterpret_cast<const float4*>(x);
float4* y4 = reinterpret_cast<float4*>(y);
```

注意：`reinterpret_cast` 不复制数据，也不改变底层字节，只改变编译器看待该地址的类型。

## 9. 为什么 `float4` 需要对齐

对齐表示地址是某个字节边界的整数倍。`float4` 的访问要求地址满足 `alignof(float4)`，在此环境中是 16-byte 对齐。

项目检查：

```cpp
const auto address = reinterpret_cast<std::uintptr_t>(pointer);
const bool aligned = address % alignof(float4) == 0;
```

`cudaMalloc` 返回的基础地址有足够对齐保证，但如果以后传入人为偏移过的指针，例如 `d_data + 1`，它可能不再满足 16-byte 对齐，所以 Device 接口仍明确验证。

## 10. `float4` 的索引单位变化

标量指针下标单位是一个 `float`：

```text
x[1] 距离 x[0] 为 4 bytes
```

向量指针下标单位是一个 `float4`：

```text
x4[1] 距离 x4[0] 为 16 bytes
```

因此：

```cpp
const int vector_count = n / 4;
```

`vector_index = 3` 代表原数组元素 `12、13、14、15`，不是只代表元素 3。

kernel 读取一个 `float4`，分别更新 `x/y/z/w` 四个分量，再写回。

## 11. `n` 不能被 4 整除怎么办

例如 `n = 11`：

```text
vector_count = 11 / 4 = 2
主体：2 个 float4 → 元素 0..7
尾部：元素 8..10
```

项目让 `start == 0` 的线程处理尾部：

```cpp
if (start == 0) {
    for (int index = vector_count * 4; index < n; ++index) {
        y[index] = alpha * x[index] + y[index];
    }
}
```

尾部最多 3 个元素，所以由一个线程处理实现简单，开销相对大规模主体很小。重点是不能忽略尾部，也不能让多个线程重复写同一尾部元素。

## 12. `float4` 为什么不保证更快

向量类型可能减少动态指令数量或改变 load/store 形式，但不会自动增加显存物理带宽。性能取决于：

- 标量版本是否已经充分合并并接近带宽上限；
- 编译器是否已对标量代码做出高效指令安排；
- 指令数量是否真是瓶颈；
- 地址对齐和数据规模；
- occupancy、缓存和架构行为。

所以正确表述是“`float4` 是一个待测假设”，不是“向量化一定更快”。

## 13. 三种实现的公平性

三种版本均：

- 计算相同 AXPY；
- 使用相同 `x`、初始 `y` 和 `alpha`；
- 与同一个 CPU reference 比较；
- 使用相同 warmup 和 measured iterations；
- 用同一个 CUDA Event 计时边界；
- 按 3 次 `float` 流量计算有效带宽（读 x、读 y、写 y）。

主要变量是地址映射或访问宽度。

## 14. 本日相关文件

| 文件 | 学习作用 |
|---|---|
| `kernels/memory_access.cu` | contiguous、strided、`float4` kernel 与 Host launch 封装 |
| `include/memory_access.h` | 三种 Device 接口 |
| `tests/test_memory_access.cu` | 空输入、尾部、非整齐长度和不同 block size 正确性 |
| `benchmarks/bench_memory_access.cu` | 公平 benchmark、有效带宽和 CSV |
| `doc/day05/day05_memory_access.csv` | 27 组原始补采结果 |

更细的地址表和结果对比见 [访问映射与实验分析](day05_access_mapping_analysis.md)。

## 15. 构建、测试与运行

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

.\build\windows-ninja\Release\test_memory_access.exe
.\build\windows-ninja\Release\bench_memory_access.exe
```

原始数据：[day05_memory_access.csv](day05_memory_access.csv)

## 16. 实验配置

| 配置 | 值 |
|---|---|
| 构建 | Release |
| GPU | RTX 2080 Ti，Compute Capability 7.5 |
| CUDA | 13.3 |
| sizes | 1000003、4194304、16777217 |
| block sizes | 64、128、256 |
| warmup | 20 |
| measured iterations | 50 |
| 汇总 | kernel Event 中位数 |
| correctness | CPU reference，`atol=1e-6`，`rtol=1e-5` |

## 17. 代表性结果

每个规模选取各模式在本轮中表现最好的 block：

| N | 模式 | block | kernel 中位数 | 有效带宽 | 正确性 |
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

27 组配置全部 PASS。

## 18. 怎样解释结果

### 18.1 连续访问明显优于跨步访问

固定 block 256 时，strided 相对 contiguous 的耗时约为：

| N | strided / contiguous |
|---:|---:|
| 1,000,003 | 2.65× |
| 4,194,304 | 3.22× |
| 16,777,217 | 4.47× |

规模越大，跨步版本劣势越明显。结果与“warp 内分散地址导致内存服务效率降低”的机制一致。

严格说，这是 benchmark 对性能现象的证明；具体多了多少事务，还需要 Nsight Compute 的硬件指标验证。

### 18.2 `float4` 只在较小规模显示明显收益

`N≈1M` 时，本轮最佳 `float4` 比最佳 contiguous 快约 22%。但在 4M 和 16M 时几乎持平，最大规模甚至约慢 1.1%。

因此本实验不支持“`float4` 总是更快”。更合理的解释是：大规模连续标量版本已经接近内存吞吐区间，减少部分指令不足以继续明显提高带宽。

### 18.3 block size 没有跨模式的统一最优值

不同规模和模式的最佳 block 不相同，且部分差距很小。block size 是实验变量，不是只靠经验固定的答案。

## 19. 常见误解

### 误解 1：每个线程访问连续数据，就一定是合并访问

合并关注的是**同一个 warp 的不同 lane 在同一条指令中访问的地址关系**，不是单个线程自己的访问历史。

### 误解 2：跨步访问一定会算错

跨步是映射方式。只要覆盖每个有效下标一次，数学结果可以完全正确，但性能较差。

### 误解 3：`float4` 会把四个 float 压缩成一个 float

不会。数据量仍是 16 bytes，只是用一个向量类型表达四个连续分量。

### 误解 4：`cudaMalloc` 对齐了，任何派生指针也都对齐

基础地址对齐不代表 `pointer + 1` 仍满足 16-byte 对齐。偏移会改变地址余数。

### 误解 5：benchmark 快就已经知道硬件根因

时间数据告诉我们现象和幅度；缓存命中、内存事务、指令数等根因需要 profiler 指标支持。

## 20. 自检题

1. warp 有多少个 lane？
2. 为什么连续地址通常比高度分散地址更高效？
3. `linear_index=67` 时，`lane` 和 `row` 分别是多少？
4. `n=19` 时有多少个完整 `float4`，尾部有几个元素？
5. 为什么 `float4` 在大规模实验中可能与标量版持平？

答案要点：32；更少且利用率更高的内存事务；`lane=3,row=2`；4 个、尾部 3 个；标量连续访问可能已接近带宽上限，瓶颈不再是指令表达宽度。

## 21. 证据边界与完成标准

当前已验证：

- 三种模式在全部 27 组配置下结果正确；
- 连续访问在所有实测大规模配置中明显优于刻意跨步访问；
- `float4` 正确处理对齐与 0～3 个尾部元素；
- `float4` 的收益随规模变化，不能泛化成必然加速。

尚未验证：

- 各版本的真实 DRAM transaction 数量；
- L1/L2 缓存命中率对结果的具体贡献；
- SASS 中标量版与 `float4` 版的实际指令差异。

只有能手算 strided 映射、解释 `float4` 对齐与尾部、并能用 CSV 的证据边界描述结论，才算完成 Day5。

下一步 Day6 会把这些全局内存原则应用到矩阵转置，并引入 shared memory、tile、同步和 bank conflict。
