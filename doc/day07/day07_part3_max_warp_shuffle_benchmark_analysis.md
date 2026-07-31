# Day 07 Part 3：Max Reduction、Warp Shuffle 与性能实验

## 1. 问题与目标

本阶段在已正确的 shared-memory sum reduction 上完成两项扩展：

1. 将合并操作替换为最大值，实现 max reduction；
2. 在只剩一个 warp 时改用寄存器 shuffle，减少 shared-memory 操作和 barrier。

实验保留 shared 基线，通过同输入、同 block、同轮数的 Release benchmark 判断
优化是否有效。

## 2. 从 Sum 推广到 Max

归约框架由两部分组成：

```text
归约结构 + 合并操作
```

sum：

```cpp
value += other;
```

max：

```cpp
value = fmaxf(value, other);
```

max 不进行累加，普通有限输入下最终结果仍是原输入中的某个 `float`，因此本实验
要求 GPU 与 CPU max 完全相等。

### 2.1 单位元

sum 越界线程写 0。max 必须写负无穷：

```cpp
shared_values[local_index] =
    global_index < n
        ? d_input[global_index]
        : -CUDART_INF_F;
```

如果填 0，纯负数输入会错误地选中 padding。

### 2.2 `fmaxf` 与 `std::max`

`fmaxf` 明确处理两个 `float`，并具有浮点最大值语义。`std::max` 是依赖比较运算
的通用 C++ 模板。对本实验的普通有限数值，两者会得到相同最大值；遇到 NaN 或
正负零时，需要先定义业务语义再决定验证规则。

## 3. Warp Shuffle 原理

`__shfl_down_sync(mask, value, offset)` 让当前 lane 读取同一 warp 中
`lane_id + offset` 的寄存器值。

32 个 lane 的归约距离依次减半：

```text
16 → 8 → 4 → 2 → 1
```

sum 辅助函数：

```cpp
for (
    int offset = warpSize / 2;
    offset > 0;
    offset /= 2
){
    value += __shfl_down_sync(
        0xffffffffu,
        value,
        offset
    );
}
```

`0xffffffffu` 表示 32 个 lane 全部参与，因此调用路径必须保证完整的第一个 warp
进入。当前 launcher 把 block 限制为 `[32,1024]` 范围内的 2 的幂。

## 4. Shared 与 Warp Kernel 的差异

以 block=256 为例，shared 基线执行：

```text
128→64→32→16→8→4→2→1
```

每轮都经过 shared memory 和 `__syncthreads()`。

warp 版本：

```text
shared memory：128→64
寄存器读取 shared[tid+32]：完成 stride=32
shuffle：16→8→4→2→1
```

最后结果在 lane 0 的寄存器 `value` 中，必须直接写：

```cpp
d_block_sums[blockIdx.x] = value;
```

不能再读取未更新的 `shared_values[0]`。

## 5. Benchmark 设计

原始数据：

[day07_reduction.csv](day07_reduction.csv)

配置：

- operation：sum、max；
- variant：shared、warp；
- N：1003、100003、1048576、16777216；
- block：32、64、128、256；
- warmup：20；
- iterations：50；
- 统计量：CUDA Event median；
- 计时范围：GPU 完整多轮归约 pipeline。

分配、H2D、最终 D2H 和 CPU reference 不计时。

## 6. 指标含义

### `pipeline_ms_median`

50 次完整 GPU pipeline 时间的中位数，单位毫秒。越低越好。中位数比“最快一次”
更能抵抗偶发抖动。

### `speedup_vs_shared`

```text
同条件 shared 时间 / 当前 variant 时间
```

- `1.0`：一样快；
- `>1.0`：当前版本更快；
- `<1.0`：当前版本退化。

### `M_elements_per_s`

每秒处理的百万个原始输入元素。越高越好。

### `effective_GB_per_s`

按每轮 global input read 和 partial output write 估算：

```text
Σ(current_count + next_count) × sizeof(float)
```

再除以 pipeline 时间。它用于公平比较同一算法版本，不是 profiler 实测的物理
DRAM 带宽，后续小轮次还可能命中 cache。

### `abs_error` 与 `correct`

- sum 使用绝对加相对误差容限；
- 普通有限 max 要求完全相等；
- 任一配置 `FAIL` 都不能进入性能结论。

## 7. 结果

64 行全部 `PASS`。warp 版本在本次所有配置中均快于同 block 的 shared 版本。

### 7.1 N=16777216

| operation | block | shared ms | warp ms | speedup |
|---|---:|---:|---:|---:|
| sum | 32 | 0.803584 | 0.606992 | 1.324× |
| sum | 64 | 0.405504 | 0.302720 | 1.340× |
| sum | 128 | 0.394400 | 0.252400 | **1.563×** |
| sum | 256 | 0.435584 | 0.282816 | 1.540× |
| max | 32 | 0.803008 | 0.608864 | 1.319× |
| max | 64 | 0.394080 | 0.303168 | 1.300× |
| max | 128 | 0.386352 | 0.251904 | **1.534×** |
| max | 256 | 0.429248 | 0.282624 | 1.519× |

### 7.2 各规模的最佳 Warp 配置

| operation | N | 最佳 block | warp ms | 有效带宽 GB/s |
|---|---:|---:|---:|---:|
| sum | 1003 | 64 | 0.006704 | 0.618 |
| sum | 100003 | 128 | 0.010272 | 39.557 |
| sum | 1048576 | 128 | 0.022608 | 188.445 |
| sum | 16777216 | 128 | 0.252400 | 270.070 |
| max | 1003 | 64 | 0.006656 | 0.623 |
| max | 100003 | 128 | 0.010240 | 39.680 |
| max | 1048576 | 128 | 0.022624 | 188.311 |
| max | 16777216 | 128 | 0.251904 | 270.602 |

## 8. 如何解释

### 已验证

- warp shuffle 没有改变本次 sum 的归约顺序结果；
- max shared/warp 与 CPU 完全一致；
- block=32 的完整 warp 边界路径正确；
- 大规模时 warp 优化收益显著，高于小规模；
- block=128 是本次大规模 workload 的最佳配置。

### 初步推断

大规模收益增加符合“减少 shared-memory 指令和 barrier”这一设计预期。小规模中
GPU 工作太少，固定开销占比较大，因此只得到约 1.06–1.13×。

block=32 在大规模下需要更多 block 和归约轮数；block=256 虽然轮数少，但可能
降低调度灵活性或受其他资源因素影响。仅凭时间不能确定具体原因。

### 仍需验证

下一次 profiler 实验可以比较：

- barrier 相关 stall；
- shared-memory load/store 指令；
- warp stall 原因；
- achieved occupancy；
- 每个 kernel 的 launch 与执行时间。

只有 profiler 证据才能把“可能因为”升级为确定的瓶颈解释。

## 9. 结论与边界

**已验证结论：** 在 RTX 2080 Ti、CUDA 13.3、sm_75、当前 float workload 上，
warp-shuffle 版本正确，并在大规模完整 GPU reduction pipeline 中取得最高
1.563× speedup。

该结论不能直接推广到：

- 其他 GPU 架构；
- double、half 或自定义数据类型；
- 包含 NaN 的 max；
- 非 2 的幂 block；
- 包含 H2D、D2H、分配和业务逻辑的端到端应用。
