# CUDA Engineering：零基础 CUDA 学习工程

这个仓库把 CUDA 学习拆成可以运行、验证和复盘的每日阶段。读者不需要预先掌握并行计算；每一天都从新概念的定义开始，再连接到源码、实验和证据。

## 学习方式

每一天建议按同一顺序进行：

```text
先读 README 中的概念和机制
  → 手算索引或数据流
  → 对照相关源码
  → Release 构建并运行正确性测试
  → 运行 benchmark / profiler
  → 根据保存的证据写结论和边界
```

不要跳过正确性验证直接解释性能。性能更快但结果错误的实现不算优化。

## 学习路线

| 天数 | 主题 | 主要产物 | 状态 |
|---|---|---|---|
| [Day 1](doc/day01/README.md) | CUDA 程序最小闭环、Host/Device、kernel 启动 | 向量加法 | 已归档 |
| [Day 2](doc/day02/README.md) | thread/block/grid、线性索引、二维索引、grid-stride loop | 向量与矩阵加法 | 已归档 |
| [Day 3](doc/day03/README.md) | 显存生命周期、RAII、同步/异步错误、Sanitizer | 公共组件与错误测试 | 已归档 |
| [Day 4](doc/day04/README.md) | 正确性、CUDA Event、端到端计时、统计与有效带宽 | Vector Add benchmark | 已归档 |
| [Day 5](doc/day05/README.md) | warp、合并访问、跨步访问、`float4` | Memory Access benchmark | 已归档 |
| [Day 6](doc/day06/README.md) | 矩阵转置、shared memory、tile、bank conflict | 三版 transpose | 已归档 |
| [Day 7](doc/day07/README.md) | 并行归约、同步、多阶段归约、warp shuffle | Sum/Max reduction | 已归档 |
| [Day 8](doc/day08/README.md) | Nsight Compute 指标、源码关联和证据链 | NCU 分析记录 | 已归档 |
| [Day 9](doc/day09/README.md) | Stream、Event、pinned memory、流水线与 Nsight Systems | Stream pipeline | 已归档 |

详细归档要求见 [归档规范](doc/ARCHIVE_CONVENTION.md)，代码命名和注释规范见 [代码规范](doc/CODE_STYLE.md)。

## 目录结构

```text
apps/          可直接运行的教学应用；负责组织完整流程
benchmarks/    性能实验；负责预热、计时、统计和 CSV
include/       对外接口与可复用公共组件
kernels/       CUDA kernel 及其 Device 启动封装
tests/         正确性、边界条件和错误注入测试
doc/dayNN/     每日概念、实验过程、结果和结论边界
scripts/       构建辅助脚本
cmake/         CTest 配置
```

`apps` 和 `kernels` 的区别：

- `kernels` 放 GPU 计算实现以及贴近 kernel 的启动封装；
- `apps` 放用户可以运行的完整程序，负责准备输入、调用接口、检查结果和打印输出；
- 一个 `.cu` 文件不一定都是 kernel。应按文件职责判断，而不是只看扩展名。

## 构建环境

当前工程配置：

- CMake 3.28 或更高；
- C++17 / CUDA C++17；
- 默认 CUDA 架构 75，对应本项目 RTX 2080 Ti；
- Windows + Visual Studio 2022 + Ninja；
- CUDA Toolkit 13.3。

如果在其他 GPU 上使用，应调整 `CMAKE_CUDA_ARCHITECTURES`，不要把架构 75 当成 CUDA 的通用固定值。

## 构建与测试

在仓库根目录执行：

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release
```

运行测试：

```powershell
& 'D:\software\VS2022\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe' `
    --preset release --output-on-failure
```

常用程序位于：

```text
build/windows-ninja/Release/
```

例如：

```powershell
.\build\windows-ninja\Release\vector_add_app.exe
.\build\windows-ninja\Release\matrix_add_app.exe
.\build\windows-ninja\Release\bench_memory_access.exe
```

## 公共组件

| 文件 | 作用 |
|---|---|
| `include/common/cuda_check.h` | 把 CUDA 错误码转换成带文件和行号的诊断 |
| `include/common/device_buffer.h` | 用 RAII 管理 Device 内存唯一所有权 |
| `include/common/launch_config.h` | 集中处理向上取整和 block size 校验 |
| `include/common/cuda_timer.h` | 用 CUDA Event 测量 stream 内 GPU 时间 |
| `include/common/compare.h` | 使用绝对/相对容差比较 CPU/GPU 结果 |
| `include/common/init_data.h` | 生成固定 seed 的可重复输入 |

## 对零基础读者的约定

- 新名词第一次出现必须给出定义，不能只给缩写；
- 索引公式必须能用小规模数字手算；
- 代码注释解释“为什么”，基础语法细节放在文档中系统说明；
- 每个性能结论必须对应环境、输入、正确性和原始数据；
- 未采集的 profiler 指标不能写成确定根因；
- “测试通过”只代表已覆盖输入，不代表所有可能输入都已证明正确。
