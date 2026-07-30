# Day 07：Reduction 第一阶段

## 1. 学习主题和阶段状态

Day 07 的总主题是 CUDA Reduction（归约）。

本次已完成第一阶段：

- 使用 shared memory 实现 block 内树形求和；
- 每个 block 输出一个部分和；
- 正确处理不足一个 block 的尾部数据；
- 在 host 端计算 grid 和动态 shared memory 大小；
- 将 GPU 部分和复制回 CPU 合并；
- 使用 CPU double reference 检查正确性；
- 覆盖空输入、warp 边界、非整除规模和纯负数数据。

Day 07 整体尚未结束。后续阶段包括：

1. 在 GPU 上继续归约部分和，直到只剩一个结果；
2. 实现 max reduction；
3. 使用 warp shuffle 优化最后一个 warp；
4. 增加 benchmark 并保存 CSV。

详细原理与实现分析见
[第一阶段专题文档](day07_part1_shared_memory_reduction_analysis.md)。

## 2. 实验目的

将包含 `N` 个 `float` 的数组按 block 分组，让每个 block 在 shared memory
中完成树形求和，并向全局显存输出一个部分和。

第一阶段的数据流为：

```text
N 个输入
→ 每个 block 搬入 shared memory
→ block 内树形归约
→ block_count 个部分和
→ 复制回 CPU
→ CPU 合并得到最终结果
```

## 3. 核心原理

### 全局索引与局部索引

```cpp
global_index = blockIdx.x * blockDim.x + threadIdx.x;
local_index = threadIdx.x;
```

- `global_index` 决定线程读取整个输入数组中的哪个元素；
- `local_index` 决定线程使用当前 block 的哪个 shared memory 位置。

每个 block 都有自己独立的一块 shared memory。不同 block 可以同时使用
`shared_values[0]`，但它们访问的不是同一块物理共享内存。

### 树形归约

以 8 个线程为例，参与计算的线程数逐轮减半：

```text
stride=4：8 个值 → 4 个部分和
stride=2：4 个值 → 2 个部分和
stride=1：2 个值 → 1 个 block 结果
```

每轮之后调用 `__syncthreads()`，保证下一轮读取到本轮已经完成的结果。

### 部分和写回

归约结束后，当前 block 的结果位于 `shared_values[0]`。每个 block 只让
`threadIdx.x == 0` 的线程执行：

```cpp
d_block_sums[blockIdx.x] = shared_values[0];
```

`blockIdx.x` 用作输出下标，使不同 block 将结果写入不同位置，避免数据竞争。

## 4. 实现步骤

1. 声明 reduction 对外接口；
2. 每个线程加载一个输入元素到 shared memory；
3. 越界线程写入求和单位元 `0.0F`；
4. 使用 stride 减半循环完成 block 内归约；
5. 由线程 0 写出当前 block 的部分和；
6. 在 host 启动函数中计算 block 数和动态 shared memory 字节数；
7. 测试程序复制部分和回 CPU，并用 double 累加；
8. 与 CPU reference 按绝对误差和相对误差判断 PASS/FAIL。

## 5. 代码文件及职责

- [`include/reduction.h`](../../include/reduction.h)：声明 block 数计算函数和
  分块求和启动接口；
- [`kernels/reduction.cu`](../../kernels/reduction.cu)：实现 shared-memory
  block reduction；
- [`tests/test_reduction.cu`](../../tests/test_reduction.cu)：生成测试数据、
  计算 CPU reference、执行 GPU reduction 并检查误差；
- [`CMakeLists.txt`](../../CMakeLists.txt)：将 reduction 加入 `cuda_ops`，
  并把测试程序与实现链接；
- [`cmake/Tests.cmake`](../../cmake/Tests.cmake)：注册 reduction 自动测试。

## 6. 构建和运行

本机 Release 构建：

```powershell
.\scripts\build.ps1 -Configuration Release
```

直接运行：

```powershell
.\build\this-pc\Release\test_reduction.exe
```

通过 CTest 运行：

```powershell
D:\python\Scripts\ctest.exe --preset release --output-on-failure
```

## 7. 指标含义与判断标准

测试输出字段：

- `case`：测试数据类型；
- `N`：输入元素数量；
- `block`：每个 block 的线程数；
- `partials`：第一轮 GPU 输出的部分和数量；
- `cpu_sum`：CPU 使用 double 顺序累加得到的参考值；
- `gpu_sum`：GPU 计算部分和后，由 CPU 使用 double 合并得到的值；
- `abs_error`：`|gpu_sum - cpu_sum|`；
- `status`：普通用例的正确性结果。

普通用例的允许误差为：

```text
1e-4 + 1e-6 × |cpu_sum|
```

## 8. 核心实验结果

测试覆盖：

- 规模：`0、1、31、32、33、100003`；
- block：`64、128、256`；
- 数据：`random、negative`。

共 36 组普通测试，全部 `PASS`。

`N=100003` 时的部分和数量：

| block | partials |
|---:|---:|
| 64 | 1563 |
| 128 | 782 |
| 256 | 391 |

这验证了向上取整的 block 数计算以及尾部越界线程填 0 的处理。

极端数据：

```text
[1e20, -1e20, 1, -2]
```

CPU 顺序累加得到 `-1`，GPU 树形加法得到 `0`。该用例标记为 `OBSERVE`，
用于说明浮点加法不满足结合律，不作为 kernel 错误。

## 9. 遇到的问题、原因和修复

### 输出变量名不一致

Kernel 参数和写回位置曾使用不同名称，会导致编译失败。最终统一为
`d_block_sums`。

### 空输入不能启动零个 block

`N=0` 时 block 数为 0，CUDA 不允许 `<<<0, ...>>>`。启动函数在 `N <= 0`
时直接返回。

### 头文件与实现函数名不一致

头文件曾写成 `reduce_sums_block_device`，实现为
`reduce_sum_blocks_device`。统一命名后解决链接问题。

### CPU 与 GPU 结果不要求逐位相等

CPU 使用顺序累加，GPU 使用树形累加，加法顺序不同。测试采用绝对误差和
相对误差，而不是直接使用 `==`。

## 10. 当前结论、局限和下一步

第一阶段证明：

- block 内 shared-memory 树形归约逻辑正确；
- 尾部不足一个 block 时仍能正确计算；
- 不同 block 大小均能产生正确部分和；
- CPU reference 和误差判断可以发现普通计算错误。

当前局限：

- 最终部分和仍由 CPU 合并；
- kernel 目前只适用于可连续减半的 block 大小；
- 尚未实现 max reduction；
- 尚未进行性能计时和 profiler 分析。

下一步使用两块 GPU 临时缓冲区交替保存结果：

```text
100003 → 391 → 2 → 1
```

直到 GPU 上只剩一个最终结果。
