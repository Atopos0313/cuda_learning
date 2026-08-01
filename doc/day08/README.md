# Day 8｜Nsight Compute：从指标到代码修改

## 今日结论

本阶段用 Nsight Compute 完成了一次可复现的 profiler 驱动实验：从 `transpose_tiled_kernel` 的高 L1/TEX 压力和 MIO Throttle 出发，定位到 `tile[32][32]` 转置读取造成的 32-way shared-memory bank conflict；改为 `tile[32][33]` 后，shared-load conflict 从 `16,252,928` 降为 `0`，独立 benchmark 中 `4096×4096` transpose 从 `0.383280 ms` 降到 `0.293168 ms`，获得约 `1.307×` 加速。

> 状态：**核心实验已验证并归档**。原计划中的 copy/reduction 固定 profiling case 未纳入本次最终结论；当前只对 transpose before/after 作性能声明。

## 学习目标

- 从多个 kernel 启动中准确筛选目标 kernel。
- 先看时间、吞吐、occupancy 和主要 warp stall，不使用“GPU 利用率低”这类模糊结论。
- 区分普通 benchmark 时间与 profiler 重放后的诊断时间。
- 使用 Raw counter 和 Source correlation 把硬件异常关联到具体代码。
- 写出“证据 → 假设 → 修改 → 结果”的完整链条。

## 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti，Compute Capability 7.5 |
| Driver | 610.62 |
| CUDA | 13.3 |
| Nsight Compute | 2026.2.1 |
| 构建 | Windows，RelWithDebInfo，`-lineinfo` |
| workload | `4096×4096` float transpose |
| block | `(32, 8, 1)` |
| grid | `(128, 128, 1)` |
| benchmark | 20 次预热，50 次计时，报告中位数 |

## 关键代码

- `kernels/transpose.cu`
  - `transpose_tiled_kernel`：`float tile[32][32]`，作为 before。
  - `transpose_padded_kernel`：`float tile[32][33]`，作为 after。
- `benchmarks/bench_transpose.cu`
  - 固定输入、正确性验证、预热和 CUDA Event 中位数计时。
- `CMakePresets.json`
  - `profile`/RelWithDebInfo 构建保留源码行号映射。

## 构建与普通 benchmark

```powershell
.\scripts\build.ps1 -Configuration RelWithDebInfo

.\build\windows-ninja\RelWithDebInfo\bench_transpose.exe |
    Tee-Object .\doc\day08\day08_transpose_benchmark.csv
```

原始数据：[day08_transpose_benchmark.csv](day08_transpose_benchmark.csv)

### 4096×4096 结果

| variant | 时间中位数 | 有效带宽 | 正确性 | 相对关系 |
|---|---:|---:|---|---:|
| naive | 0.819072 ms | 163.87 GB/s | PASS | 基线 |
| tiled | 0.383280 ms | 350.18 GB/s | PASS | 相对 naive `2.137×` |
| padded | 0.293168 ms | 457.82 GB/s | PASS | 相对 tiled `1.307×` |

所有 CSV 用例均为 `PASS`，transpose 只重排已有 float，没有算术舍入，因此 `max_abs_error=0`、`max_rel_error=0` 符合预期。

小矩阵的固定启动开销占主导，不适合用 GB/s 判断 kernel 上限；性能结论以大矩阵为主。

## Nsight Compute 筛选

对 `bench_transpose` 中的 tiled/padded 版本分别筛选。每种尺寸启动：

```text
1次正确性 + 20次预热 + 50次计时 = 71次
```

`4096×4096` 是第 5 种尺寸，因此：

```text
launch-skip = 4 × 71 = 284
```

示例：

```powershell
$ncu = (Get-Command ncu).Source

& $ncu `
    --set detailed `
    --section WarpStateStats `
    --section SchedulerStats `
    --kernel-name-base function `
    --kernel-name transpose_tiled_kernel `
    --launch-skip 284 `
    --launch-count 1 `
    --export .\reports\day08_before `
    .\build\windows-ninja\RelWithDebInfo\bench_transpose.exe
```

after 将 kernel 名和输出改为：

```text
transpose_padded_kernel
reports/day08_after
```

报告清单见：[reports/day08_analysis.md](../../reports/day08_analysis.md)。

## 初学阶段需要掌握的指标

| 指标 | 含义 | 本实验中的判断方法 |
|---|---|---|
| Duration | NCU 单次被采集 kernel 的诊断时间 | 只比较相同采集配置；真实性能以普通 benchmark 为准 |
| Memory Throughput | 内存系统相对峰值的最高利用程度 | 与 Compute、L1/TEX、DRAM 分项一起判断瓶颈位置 |
| Compute Throughput | SM 计算/执行资源利用程度 | transpose 很低是正常现象，不代表代码错误 |
| Achieved Occupancy | 实际活跃 warp 相对硬件上限的比例 | 只作上下文；高 occupancy 不保证快 |
| Warp Stall | warp 无法继续发射指令的原因 | before 看 MIO Throttle，after 看 Long Scoreboard |
| Bank Conflicts | shared-memory 请求因同 bank 多地址访问产生的额外拆分 | before 高、after 为 0 是直接因果证据 |

## 核心结果

| 指标 | tiled before | padded after | 变化 |
|---|---:|---:|---:|
| NCU Duration | 373.95 μs | 280.16 μs | -25.08% |
| Compute Throughput | 12.14% | 15.20% | +25.19%（相对变化） |
| DRAM Throughput | 55.42% | 72.48% | +17.06 个百分点 |
| L1/TEX Throughput | 96.36% | 16.89% | -82.47%（相对变化） |
| Memory Throughput | 361.47 GB/s | 475.38 GB/s | +31.51% |
| Achieved Occupancy | 92.71% | 97.63% | +4.92 个百分点 |
| Warp Cycles/Issued Instruction | 124.14 | 98.04 | -21.03% |
| Shared-load conflicts | 16,252,928 | 0 | 完全消除 |
| Shared-store conflicts | 0 | 0 | 保持无冲突 |
| Shared-load wavefronts | 16,777,216（由计数推导） | 524,288 | 32份拆分恢复为1份 |

下面是 Nsight Compute 的同配置 Compare 视图。蓝色为 padded after，绿色为 tiled before：

![Nsight Compute before/after 时间与吞吐对比](images/07_after_compare_speed_memory.png)

## 证据链

```text
证据：
L1/TEX = 96.36%
MIO Throttle = 48.2 cycles，约占发射间隔38.8%
shared-load conflicts = 16,252,928

假设：
tile[32][32]跨行读取时，一个warp的32个线程命中同一bank

修改：
tile[32][32] → tile[32][33]

结果：
shared-load conflict → 0
每个warp shared load从32个wavefront恢复为1个
L1/TEX压力大幅下降
普通benchmark获得1.307×加速

瓶颈转移：
主要stall从MIO Throttle转为Long Scoreboard
DRAM Throughput升至72.48%
```

## 为什么 padding 有效

shared memory 有 32 个 bank。before 中固定列跨行读取：

```text
tile[0][c], tile[1][c], ..., tile[31][c]
```

每行步长为 32 个 float，对 32 取模后落入同一个 bank。一个 warp 请求被拆成 32 个 wavefront，额外冲突数为：

```text
(4096×4096 / 32) × 31 = 16,252,928
```

padding 后每行步长为 33：

```text
bank = (row×33 + c) mod 32
```

不同 row 会落到不同 bank，因此冲突为 0。增加的 shared memory 只有每 block 128 字节，没有改变由 warp 数决定的理论 occupancy 上限。

## Source correlation

RelWithDebInfo 的 `-lineinfo` 让 NCU 将 CUDA 源码映射到 SASS：

- `tile[threadIdx.y + offset][threadIdx.x] = input[...]` 对应合并的全局读取和无冲突 shared store。
- `output[...] = tile[threadIdx.x][threadIdx.y + offset]` 对应 shared load 与全局写出；bank conflict 位于该转置读取布局。
- `__syncthreads()` 不能因 Barrier stall 被直接删除，它保证所有 tile 写入完成后再读取。

## 问题与处理

### `ERR_NVGPUCTRPERM`

首次采集时驱动拒绝访问 GPU performance counters。开启 NVIDIA Developer Settings 中的性能计数器访问权限后，报告成功生成。

### detailed 集没有直接 bank-conflict 数值

通过自定义 metrics 补采：

```text
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
```

## 结论边界

### 已验证

- kernel 筛选确实命中 `(128,128,1)×(32,8,1)` 的 4096² tiled/padded transpose。
- tiled 的 shared load 是完整的 32-way bank conflict；padded 将其消除。
- 普通 benchmark 与 NCU 指标都支持 padded 更快。
- occupancy 不是 before 的根因；padding 也没有降低理论 occupancy。
- 优化后主要限制转移到全局内存/DRAM 路径。

### 未完成或不推广

- 未完成原计划中的 copy/reduction 固定 profiling 对照，因此不对这两类 kernel 作 Day 8 性能结论。
- NCU 报告时间包含计数器采集与重放语境，不能替代普通 benchmark。
- 结论来自 RTX 2080 Ti、CUDA 13.3 和当前 workload，其他 GPU/尺寸需重新测量。
- 本阶段未运行 Sanitizer/Synccheck。

## 专题文档

- [Nsight Compute 驱动的 transpose bank conflict 实验](day08_ncu_transpose_analysis.md)
- [原始 benchmark CSV](day08_transpose_benchmark.csv)
- [NCU 报告索引](../../reports/day08_analysis.md)
- [实验截图目录](images/)

## 截图证据索引

| 图片 | 内容 |
|---|---|
| [01_before_overview.png](images/01_before_overview.png) | before 的 Speed of Light 与 Memory Workload |
| [02_before_warp_state.png](images/02_before_warp_state.png) | before 的 MIO Throttle 主停顿 |
| [03_before_occupancy.png](images/03_before_occupancy.png) | before occupancy 与资源限制 |
| [04_initial_bank_search.png](images/04_initial_bank_search.png) | 首次搜索只得到 bank 吞吐指标 |
| [05_before_bank_conflicts.png](images/05_before_bank_conflicts.png) | before 的精确 shared-load conflict |
| [06_after_bank_conflicts.png](images/06_after_bank_conflicts.png) | after conflict 归零与理想 wavefront |
| [07_after_compare_speed_memory.png](images/07_after_compare_speed_memory.png) | before/after 时间和吞吐对比 |
| [08_after_occupancy.png](images/08_after_occupancy.png) | padding 后 occupancy 没有下降 |
| [09_after_workload_distribution.png](images/09_after_workload_distribution.png) | after 的 DRAM workload distribution |
| [10_after_warp_state_compare.png](images/10_after_warp_state_compare.png) | stall 从 MIO 转移到 Long Scoreboard |
| [11_source_correlation.png](images/11_source_correlation.png) | CUDA 源码与 SASS 的 source correlation |

## 下一步

Day 9 进入 Stream、Pinned Memory 与拷贝计算重叠。开始前不再扩展 Day 8 的 Raw/SASS 指标；只在新实验需要时复用本日形成的“证据 → 假设 → 修改 → 验证”方法。
