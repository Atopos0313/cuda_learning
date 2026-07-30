# Day 07 Part 1：Shared Memory 分块归约分析

## 1. 为什么需要归约

向量加法中，每个输出元素彼此独立，一个线程可以直接负责一个元素。求和不同：
所有输入最终共同生成一个结果，线程之间必须合并中间结果。

CPU 的普通求和循环具有明显的串行依赖：

```text
result = result + input[0]
result = result + input[1]
result = result + input[2]
...
```

CUDA reduction 将线性累加改成树形合并，使一轮中的多组加法可以并行执行。

## 2. 从 8 个元素理解树形归约

初始 shared memory：

```text
下标：0 1 2 3 4 5 6 7
数据：1 2 3 4 5 6 7 8
```

第一轮 `stride=4`：

```text
T0: shared[0] += shared[4]
T1: shared[1] += shared[5]
T2: shared[2] += shared[6]
T3: shared[3] += shared[7]
```

前四个位置变为：

```text
[6, 8, 10, 12]
```

第二轮 `stride=2`：

```text
T0: shared[0] += shared[2]
T1: shared[1] += shared[3]
```

前两个位置变为：

```text
[16, 20]
```

第三轮 `stride=1`：

```text
T0: shared[0] += shared[1]
```

最终：

```text
shared[0] = 36
```

## 3. 为什么需要同步

下一轮依赖上一轮写出的数据。CUDA 不保证同一个 block 内所有线程以完全相同
速度执行，因此每轮结束后必须使用：

```cpp
__syncthreads();
```

同步必须位于条件分支之外，让 block 内所有线程都能到达：

```cpp
if (local_index < stride) {
    // 部分线程进行加法
}

__syncthreads();
```

如果只有部分线程执行 `__syncthreads()`，可能造成 block 永久等待。

## 4. 为什么越界线程写 0

输入规模通常不是 block 大小的整数倍。例如 10 个元素、每个 block 4 个线程，
最后一个 block 只有两个有效输入：

```text
block 2：[9, 10, ?, ?]
```

求和的单位元是 0，因此越界线程写入 0：

```text
block 2：[9, 10, 0, 0]
```

归约结果仍然是 19，不需要为最后一个 block 单独编写算法。

## 5. 为什么每个 block 只能先得到部分和

`__syncthreads()` 只能同步同一 block 内的线程，不能在一个普通 kernel 中完成
所有 block 的全局同步。因此第一轮只能得到：

```text
block 0 → d_block_sums[0]
block 1 → d_block_sums[1]
block 2 → d_block_sums[2]
...
```

当前测试将这些部分和复制回 CPU 合并。下一阶段会由 host 继续启动相同 kernel，
在 GPU 上将部分和逐轮缩小。

## 6. 动态 shared memory

Kernel 中的：

```cpp
extern __shared__ float shared_values[];
```

表示 shared memory 大小不在编译期固定，而是在 kernel 启动时通过第三个配置
参数指定：

```cpp
kernel<<<blocks, threads, shared_bytes, stream>>>(...);
```

当前一个线程需要保存一个 `float`，因此：

```text
shared_bytes = threads_per_block × sizeof(float)
```

该大小是每个 block 的 shared memory 大小。

## 7. 浮点非结合性

数学实数满足：

```text
(a + b) + c = a + (b + c)
```

有限精度浮点数不一定满足。CPU 顺序循环与 GPU 树形归约使用不同的加法顺序，
所以低位误差不同。

极端数据：

```text
[1e20, -1e20, 1, -2]
```

CPU 当前顺序得到 `-1`，GPU 树形顺序得到 `0`。这说明 reduction 的正确性
检查必须使用合理容差，并区分正常舍入误差与真正的索引、同步或边界错误。

## 8. 第一阶段结论

本阶段掌握了：

- 全局索引负责读取输入；
- 局部索引负责访问当前 block 的 shared memory；
- shared memory 属于 block，不属于整个 grid；
- stride 减半形成树形归约；
- 每轮之间必须进行 block 内同步；
- 线程 0 将 block 结果写入 `d_block_sums[blockIdx.x]`；
- 第一轮部分和数量等于向上取整的 block 数；
- 浮点 reduction 应使用误差容限验证。

下一阶段将复用现有 kernel，并通过 ping-pong buffer 完成全 GPU 多轮归约。
