# Day 6 第二部分：Shared-memory Tiled 转置分析

> 归档位置：`doc/day06/`；原始数据见同目录 `day06_transpose.csv`。

## 1. 本阶段解决什么问题

Naive 转置的问题是：

```text
全局显存读取：连续、合并
全局显存写入：跨步、不合并
```

Tiled 版本的目标不是减少矩阵元素，也不是减少算法必需的数据量，而是借助
shared memory 改变数据排列顺序：

```text
从全局显存连续读取
→ 在 block 内写入 shared-memory tile
→ 交换 tile 的行列索引
→ 向全局显存连续写出
```

因此 shared memory 在这里相当于一个位于 SM 上的“数据重排站”。

## 2. 为什么 tile 是 `32 × 32`

一个 warp 有 32 个线程。将 tile 的 x 方向设为 32，可以让同一个 warp 的
线程在执行一条全局 load/store 指令时处理连续的 32 个 `float`：

```text
lane 0 处理第 0 列
lane 1 处理第 1 列
...
lane 31 处理第 31 列
```

32 个 `float` 总计：

```text
32 × 4 Byte = 128 Byte
```

所以 `32` 与 warp 宽度自然对应，便于构造连续访问。

shared-memory tile 的大小为：

```text
32 × 32 × 4 Byte = 4096 Byte
```

即每个 block 使用 4 KiB shared memory。

## 3. 为什么 tile 是 `32 × 32`，block 却是 `32 × 8`

当前配置：

```cpp
dim3 block(32, 8);
```

一个 block 只有：

```text
32 × 8 = 256 个线程
```

但 tile 有：

```text
32 × 32 = 1024 个元素
```

因此每个线程处理 4 个元素。循环的 offset 为：

```text
0、8、16、24
```

对于 `threadIdx.y = 3` 的线程，它依次处理 tile 的：

```text
第 3 行
第 11 行
第 19 行
第 27 行
```

所以：

```text
256 个线程 × 每线程 4 个元素 = 1024 个元素
```

如果使用 `32 × 32` 个线程，就需要 1024 个线程，已经达到常见的每 block
最大线程数，而且没有必要为每个元素都配置一个独立线程。`32 × 8` 让一个
线程复用四次，同时保留 x 方向的 32 个连续线程。

## 4. 第一阶段：连续读入 shared memory

输入坐标为：

```cpp
input_col = blockIdx.x * 32 + threadIdx.x;
input_row = blockIdx.y * 32 + threadIdx.y;
```

循环中执行：

```cpp
tile[threadIdx.y + offset][threadIdx.x]
    = input[(input_row + offset) * width + input_col];
```

对于同一个 warp：

- `threadIdx.y` 相同；
- `offset` 相同；
- `threadIdx.x` 从 0 到 31。

所以它读取：

```text
input[row][0]
input[row][1]
...
input[row][31]
```

全局读取地址连续，能够合并。

写入 shared memory 时访问：

```text
tile[固定行][0..31]
```

这一阶段的 shared-memory 写入也是按列连续展开的。

## 5. 为什么必须调用 `__syncthreads()`

一个线程后面读取的 shared-memory 元素可能是另一个线程刚刚写入的。

例如某线程写入：

```text
tile[5][20]
```

转置读取时，负责输出的可能是另一个线程，它会从：

```text
tile[threadIdx.x][threadIdx.y + offset]
```

读取这个位置。

没有同步时，不能保证所有线程已经完成写入；某些线程可能提前读到尚未更新的
shared memory。因此需要：

```cpp
__syncthreads();
```

它保证：

```text
block 中所有线程完成 tile 写入
→ 所有线程才能继续读取 tile
```

这里同步位于边界判断之外，所以 block 内所有线程都会到达同步点，不会因为
部分线程跳过同步而造成死锁。

## 6. 第二阶段：交换 tile 坐标并连续写出

输出坐标使用：

```cpp
output_col = blockIdx.y * 32 + threadIdx.x;
output_row = blockIdx.x * 32 + threadIdx.y;
```

注意 `blockIdx.x` 和 `blockIdx.y` 交换了职责：

```text
输入 tile 坐标：(blockIdx.y, blockIdx.x)
输出 tile 坐标：(blockIdx.x, blockIdx.y)
```

这对应矩阵转置时：

```text
输入的行块变成输出的列块
输入的列块变成输出的行块
```

输出时读取 shared memory：

```cpp
tile[threadIdx.x][threadIdx.y + offset]
```

这里也交换了 tile 内的行列索引。

最终，全局写入为：

```cpp
output[(output_row + offset) * height + output_col]
```

对同一个 warp 而言：

- `output_row + offset` 固定；
- `output_col` 随 `threadIdx.x` 连续变化。

所以一个 warp 写入输出矩阵同一行的连续 32 个元素：

```text
output[row][0]
output[row][1]
...
output[row][31]
```

这正是 tiled 版本相对 naive 版本的核心变化：

```text
Naive：连续读取 + 跨步写入
Tiled：连续读取 + 连续写入
```

## 7. 为什么 grid 的 Y 方向按 32 而不是 8 计算

虽然 block 在 Y 方向只有 8 个线程，但每个线程通过循环又处理了 4 行：

```text
8 行/轮 × 4 轮 = 32 行
```

因此一个 tiled block 实际覆盖 `32 × 32` 个输入元素，grid 应按 tile 尺寸
计算：

```cpp
grid.x = ceil(width / 32)
grid.y = ceil(height / 32)
```

这与 naive Kernel 不同。Naive Kernel 没有 Y 方向循环，一个 block 只覆盖
`32 × 8` 个元素，所以 naive 的 `grid.y` 必须按 8 计算。

## 8. Tiled 原始结果

| height × width | Naive 时间 | Tiled 时间 | Naive 带宽 | Tiled 带宽 | 加速比 |
|---:|---:|---:|---:|---:|---:|
| 3 × 5 | 0.004128 ms | 0.004288 ms | 0.03 GB/s | 0.03 GB/s | 0.96× |
| 17 × 19 | 0.004320 ms | 0.004496 ms | 0.60 GB/s | 0.57 GB/s | 0.96× |
| 1003 × 769 | 0.045616 ms | 0.025520 ms | 135.27 GB/s | 241.79 GB/s | 1.79× |
| 2048 × 2048 | 0.208912 ms | 0.110928 ms | 160.62 GB/s | 302.49 GB/s | 1.88× |
| 4096 × 4096 | 0.816448 ms | 0.421376 ms | 164.39 GB/s | 318.52 GB/s | 1.94× |

加速比按以下公式计算：

```text
加速比 = Naive 时间 / Tiled 时间
```

加速比大于 1 才表示 tiled 更快。

## 9. 正确性怎样判断

Tiled 版本五种尺寸全部：

```text
correct = PASS
max_abs_error = 0
max_rel_error = 0
first_mismatch = none
```

这说明：

1. tile 内部行列交换正确；
2. block 坐标交换正确；
3. 非方阵的输出索引正确；
4. 边缘 tile 的范围检查有效；
5. `__syncthreads()` 后的数据读取没有产生可见错误。

因为转置只复制数据，不进行浮点计算，所以这里应该精确一致，零误差是合理
要求。

## 10. 小矩阵为什么反而略慢

`3 × 5` 和 `17 × 19` 中，tiled 的加速比约为 `0.96×`，即略慢约 4%。

原因是小矩阵的数据量不足以补偿 tiled 版本增加的工作：

- 分配并访问 shared memory；
- 执行 4 轮循环；
- 执行边界判断；
- 执行一次 `__syncthreads()`；
- 大量线程由于矩阵太小而不处理有效元素。

例如 `3 × 5` 只有 15 个有效元素，但仍启动一个包含 256 个线程的 block。

因此：

> shared memory 不是免费加速。数据太小时，额外同步和控制开销可能超过它
> 节省的全局内存事务。

小矩阵仍主要用于验证正确性。

## 11. 大矩阵为什么明显变快

大矩阵时，Naive 的跨步写入会反复制造低效全局内存事务。Tiled 版本先把数据
放入片上 shared memory，再以连续地址写回，显著减少了全局写入侧的浪费。

结果随规模增大而更加清楚：

```text
1003 × 769：1.79×
2048 × 2048：1.88×
4096 × 4096：1.94×
```

`4096 × 4096` 的时间从：

```text
0.816448 ms → 0.421376 ms
```

减少约 48.4%，有效带宽从：

```text
164.39 GB/s → 318.52 GB/s
```

提高约 93.8%。

随着矩阵增大：

- 固定启动和同步开销被更多元素摊薄；
- 可并行 tile 数量增多；
- GPU 更容易进入稳定吞吐状态；
- 合并全局写入带来的收益占据主导。

## 12. 不规则尺寸为什么低于大方阵

`1003 × 769` 的 tiled 带宽为 `241.79 GB/s`，低于两个大方阵。

它需要：

```text
grid.x = ceil(769 / 32) = 25
grid.y = ceil(1003 / 32) = 32
```

边缘 tile 中存在无效线程，仍要参与循环和同步。同时这个规模的总执行时间只有
约 25.5 微秒，固定开销仍占有一定比例。

因此其带宽较低是合理现象，不能直接归因于算法错误。因为正确性为 PASS，而且
矩阵增大后带宽继续上升并趋于稳定。

## 13. 为什么 `318.52 GB/s` 仍未接近显存峰值

当前 tiled 版本已经解决全局显存的跨步写问题，但它还存在一个新的瓶颈：
shared-memory bank conflict。

shared memory 可以简化理解为 32 个 bank。对 32 位 `float`，连续元素通常
映射到连续 bank。

当前声明：

```cpp
float tile[32][32];
```

按行主序展开后：

```text
tile[row][col] 的 word 下标 = row × 32 + col
bank ≈ (row × 32 + col) mod 32
```

写入 tile 时，一个 warp 访问：

```text
tile[固定 row][threadIdx.x]
```

bank 为：

```text
(固定 row × 32 + threadIdx.x) mod 32
= threadIdx.x
```

32 个线程落入 32 个不同 bank，基本没有冲突。

但转置读取时，一个 warp 访问：

```text
tile[threadIdx.x][固定 col]
```

bank 为：

```text
(threadIdx.x × 32 + 固定 col) mod 32
= 固定 col
```

32 个线程访问的是不同地址，却全部映射到同一个 bank。这不是同地址广播，
而是严重的 bank conflict，需要被拆分处理。

所以当前版本的状态是：

```text
全局读：合并
全局写：合并
shared 写：无明显冲突
shared 转置读：严重 bank conflict
```

它仍能比 naive 快，是因为片上 shared-memory 冲突的代价，依然小于 naive
版本大量分散的全局显存写事务。但 bank conflict 限制了进一步提升。

## 14. 为什么用理论带宽比较时必须谨慎

RTX 2080 Ti 的参考理论显存带宽是 `616 GB/s`。最大尺寸的 tiled 有效带宽：

```text
318.52 / 616 ≈ 51.7%
```

这说明 tiled 比 naive 的约 `26.7%` 有了大幅改善。

但 `51.7%` 仍不能直接称为精确的显存利用率：

- `616 GB/s` 是理论峰值；
- `318.52 GB/s` 是按有用读写量计算的有效带宽；
- Kernel 还执行 shared-memory 访问、同步、地址计算和边界判断；
- 是否真的由 bank conflict 主导，最终应由 profiler 指标进一步验证。

benchmark 数据与代码访问模式强烈支持 bank conflict 判断，但目前还没有
profiler 数据，因此不应伪造精确的冲突次数或周期损失。

## 15. 本阶段结论

1. Tiled 版本在所有测试尺寸上结果正确。
2. 小矩阵因固定开销和同步成本略慢，shared memory 并非对所有规模都有收益。
3. 大矩阵达到 `1.79～1.94×` 加速，证明连续全局写入显著减少了内存事务浪费。
4. 最大尺寸有效带宽从 `164.39 GB/s` 提升到 `318.52 GB/s`，但仍有优化空间。
5. 当前 `tile[32][32]` 在转置读取时产生严重 shared-memory bank conflict。
6. 下一阶段应把 tile 的第二维从 32 改为 33，通过 padding 改变 bank 映射，
   然后重新 benchmark，判断消除冲突能带来多少实际收益。

## 16. 一句话总结

> Tiled 转置用 shared memory 把“连续读、跨步写”改造成“连续读、连续写”，
> 因而在大矩阵上接近 2 倍加速；但 `32 × 32` shared tile 的转置读取会让
> 一个 warp 的不同地址落到同一 bank，下一步需要 padding 消除 bank conflict。
