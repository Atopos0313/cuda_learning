# Day 6：Shared Memory 与矩阵转置优化

> Day 6 通过 Naive、Tiled 和 Padded 三个矩阵转置 Kernel，建立了从全局显存合并访问到 shared-memory bank conflict 优化的完整实验链。最大矩阵上，最终版本相对 Naive 加速 `2.81×`，有效带宽从 `163.84 GB/s` 提升到 `459.93 GB/s`，且全部结果精确正确。

## 1. 本日目标

1. 理解二维矩阵转置的一维下标推导；
2. 观察 Naive 转置为什么出现非合并全局写入；
3. 使用 shared memory 重排数据，使全局读写都连续；
4. 理解 `__syncthreads()` 的作用和使用位置；
5. 理解 shared-memory bank 与 bank conflict；
6. 使用 `32 × 33` padding 消除列读取冲突；
7. 通过统一 benchmark 验证正确性和性能收益。

## 2. 文档与数据索引

| 文件 | 内容 |
|---|---|
| [第一部分：Naive 分析](day06_part1_naive_analysis.md) | 指标解释、转置索引、非合并写入和性能基线 |
| [第二部分：Tiled 分析](day06_part2_tiled_analysis.md) | `32×32` tile、`32×8` block、同步、全局读写合并和 bank conflict |
| [第三部分：Padded 分析](day06_part3_padded_analysis.md) | bank 映射、`32×33` padding、三版本性能对比和实验局限 |
| [最终原始数据](day06_transpose.csv) | Naive、Tiled、Padded 三版本，共15组结果 |

## 3. 当天涉及的代码

### 核心代码

| 文件 | 作用 | Day 6 内容 |
|---|---|---|
| [include/transpose.h](../../include/transpose.h) | 对外接口 | 声明 Naive、Tiled、Padded 三个 device wrapper |
| [kernels/transpose.cu](../../kernels/transpose.cu) | CUDA Kernel 实现 | 转置索引、shared-memory tile、同步、padding |
| [benchmarks/bench_transpose.cu](../../benchmarks/bench_transpose.cu) | 正确性与性能 benchmark | CPU reference、三版本分发、预热、CUDA Event 计时、CSV 输出 |
| [CMakeLists.txt](../../CMakeLists.txt) | 构建配置 | 编译 `transpose.cu` 并生成 `bench_transpose` |

### 复用的公共组件

| 文件 | 用途 |
|---|---|
| [compare.h](../../include/common/compare.h) | 比较 GPU 输出和 CPU reference |
| [cuda_check.h](../../include/common/cuda_check.h) | 检查 CUDA API 与 Kernel 启动错误 |
| [cuda_timer.h](../../include/common/cuda_timer.h) | 使用 CUDA Event 测量 Kernel 时间 |
| [device_buffer.h](../../include/common/device_buffer.h) | 管理 device memory 生命周期 |
| [init_data.h](../../include/common/init_data.h) | 生成可重复的输入数据 |

## 4. 知识点总览

| 知识点 | 核心含义 | 在本实验中的作用 |
|---|---|---|
| 行主序 | 二维数组按行连续存储 | 推导输入、输出线性下标 |
| Warp | NVIDIA GPU 的32线程执行单位 | 决定连续32个线程的访存模式 |
| 合并访存 | 同一 warp 访问连续、对齐地址 | 降低全局内存事务浪费 |
| Shared memory | 同一 block 共享的片上临时空间 | 在 block 内重排矩阵 tile |
| `__syncthreads()` | block 级屏障 | 保证所有 tile 写入完成后再读取 |
| Shared-memory bank | shared memory 的并行访问通道 | 决定同一 warp 的 shared 访问能否并行 |
| Bank conflict | 不同地址落入同一 bank，需要拆分处理 | 限制普通 Tiled 版本性能 |
| Padding | 增加不用的物理存储改变地址跨度 | 将列访问分散到不同 bank |
| 有效带宽 | 算法有效读写字节数除以时间 | 评价内存型 Kernel 的吞吐 |
| 中位数 | 排序后取中间样本 | 降低偶发抖动对计时结果的影响 |

## 5. 转置下标

输入矩阵为 `height × width`，行主序输入下标：

```cpp
input_index = row * width + col;
```

转置后：

```text
新行 = 原 col
新列 = 原 row
输出每行元素数 = 原 height
```

所以输出下标：

```cpp
output_index = col * height + row;
```

Naive Kernel 的同一 warp：

```text
input[row][0..31]：连续读取
output[0..31][row]：跨步写入
```

## 6. 三阶段优化原理

### 6.1 Naive

```text
全局读取：连续、合并
全局写入：跨步、不合并
```

大矩阵有效带宽约为 `164 GB/s`。

### 6.2 Tiled

使用：

```cpp
__shared__ float tile[32][32];
```

执行流程：

```text
连续读取 input
→ tile[y][x]
→ __syncthreads()
→ tile[x][y]
→ 连续写入 output
```

全局读写都合并，但转置读取 `tile[x][y]` 时产生严重 bank conflict。最大矩阵
有效带宽提升到 `318.76 GB/s`。

### 6.3 Padded

改为：

```cpp
__shared__ float tile[32][33];
```

`32×32` 的 bank 映射：

```text
bank = (row × 32 + col) mod 32
     = col
```

同一列落入同一 bank。

`32×33` 的 bank 映射：

```text
bank = (row × 33 + col) mod 32
     = (row + col) mod 32
```

同一列的不同行被分散到不同 bank。每个 block 只增加：

```text
32 × 4 Byte = 128 Byte
```

最大矩阵有效带宽提升到 `459.93 GB/s`。

## 7. 为什么使用 `32 × 8` 个线程

逻辑 tile 包含：

```text
32 × 32 = 1024 个元素
```

block 包含：

```text
32 × 8 = 256 个线程
```

每个线程通过：

```text
offset = 0、8、16、24
```

处理4个元素：

```text
256 × 4 = 1024
```

x方向保持32个线程，使一个 warp 能够处理连续的32个 `float`。

## 8. Benchmark 方法

构建 Release：

```powershell
cmake --build --preset release
```

运行并以 UTF-8 保存最终结果：

```powershell
$result = .\build\windows-ninja\Release\bench_transpose.exe
$result
$result | Set-Content -Encoding utf8 .\doc\day06\day06_transpose.csv
```

每个版本：

```text
先运行一次并与 CPU reference 比较
→ 预热20次
→ 正式计时50次
→ 取 Kernel 时间中位数
```

## 9. 指标解释

| 指标 | 含义 | 判断 |
|---|---|---|
| `height/width` | 输入矩阵行列数 | 用于覆盖方阵、非方阵和边界尺寸 |
| `elements` | `height × width` | 问题规模 |
| `warmup` | 正式计时前执行次数 | 本实验为20 |
| `iters` | 正式计时样本数 | 本实验为50 |
| `kernel_ms_median` | Kernel 时间中位数 | 相同工作下越低越好 |
| `GB_per_s` | 算法有效读写带宽 | 相同工作下越高越好 |
| `correct` | 是否与 CPU reference 一致 | 必须为 `PASS` |
| `max_abs_error` | 最大绝对误差 | 本实验应严格为0 |
| `max_rel_error` | 最大相对误差 | 本实验应严格为0 |
| `first_mismatch` | 首个不一致下标 | 正确时为 `none` |

有效带宽公式：

```text
每个 float：读取4 Byte + 写入4 Byte = 8 Byte

GB/s = elements × 8 / kernel_seconds / 1e9
```

它是算法有效带宽，不等于显存总线的真实字节流量。

## 10. 最终结果

### 大矩阵有效带宽

| 规模 | Naive | Tiled | Padded |
|---:|---:|---:|---:|
| 1003 × 769 | 134.75 GB/s | 239.09 GB/s | 495.70 GB/s |
| 2048 × 2048 | 164.08 GB/s | 303.41 GB/s | 496.37 GB/s |
| 4096 × 4096 | 163.84 GB/s | 318.76 GB/s | 459.93 GB/s |

### 加速比

| 规模 | Tiled → Padded | Naive → Padded |
|---:|---:|---:|
| 1003 × 769 | 2.07× | 3.68× |
| 2048 × 2048 | 1.64× | 3.03× |
| 4096 × 4096 | 1.44× | 2.81× |

全部15组结果：

```text
correct = PASS
max_abs_error = 0
max_rel_error = 0
first_mismatch = none
```

## 11. 如何解读结果

### 小矩阵

`3 × 5` 和 `17 × 19` 的总时间约4微秒，主要受 Kernel 启动、调度、同步和
计时固定开销影响。它们用于验证正确性，不适合评价显存带宽。

### 大矩阵

Naive 在大矩阵上稳定于约 `164 GB/s`，证明跨步全局写入是稳定瓶颈。

Tiled 通过合并全局写入提升到约 `239～319 GB/s`。

Padded 进一步消除 shared-memory bank conflict，提升到约 `460～496 GB/s`。

`4096 × 4096` 低于 `2048 × 2048` 的具体原因尚未由 profiler 确认，可能与
更大工作集、缓存、地址转换、GPU状态或正常测量波动有关，不能只凭现有数据
归因于某一个硬件因素。

## 12. 本日遇到的问题与修复

| 问题 | 原因 | 修复 | 可复用经验 |
|---|---|---|---|
| Naive 可能漏算Y方向数据 | Naive block 的Y方向只有8个线程，却错误按32计算 grid | Naive `grid.y` 按 `block.y` 计算 | grid 覆盖范围必须按 Kernel 实际处理量推导 |
| Device wrapper 接收 stream 但未使用 | Kernel 使用默认 stream 启动 | 使用 `<<<grid, block, 0, stream>>>` | 包装函数参数必须真正传递到底层启动配置 |
| Tiled 全局读写已合并但仍不够快 | `tile[32][32]` 按列读取产生 bank conflict | 使用 `tile[32][33]` | 优化全局显存后还要检查片上存储访问模式 |
| 小矩阵 GB/s 极低 | 固定启动开销相对数据量过大 | 小尺寸只判断正确性，大尺寸判断吞吐 | 性能指标必须结合问题规模解释 |

## 13. 能证明与不能证明的内容

### 当前实验支持

1. 三个 Kernel 在已测尺寸上转置正确；
2. 合并全局写入能显著改善矩阵转置；
3. `32×33` 比 `32×32` 更适合当前列读取方式；
4. Bank conflict 是普通 Tiled 版本的重要瓶颈；
5. 数据布局优化可以在不减少数学工作量的情况下显著提高性能。

### 仍需进一步验证

1. 精确的 shared-memory 冲突次数；
2. 每个版本的真实 DRAM 流量；
3. 缓存对中等尺寸结果的影响；
4. `4096 × 4096` 带宽下降的具体原因；
5. 其他 tile 与 block 配置是否更优。

这些问题需要重复实验或 Nsight Compute 硬件计数器支持。

## 14. 本日总结

Day 6 建立了两层访存优化方法：

```text
第一层：让全局显存访问连续、合并
第二层：让 shared-memory 访问分散到不同 bank
```

最终版本没有改变转置算法，只通过数据重排和128 Byte/block的padding，使最大
矩阵相对 Naive 加速 `2.81×`。

一句话总结：

> Shared memory 的价值不仅是“更快的内存”，更重要的是它允许线程块在片上
> 重排数据；而 padding 说明，即使数据位于片上，访问布局仍然决定并行效率。
