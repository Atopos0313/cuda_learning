# Day 1｜建立可重复构建与首个 Kernel 闭环

> 补档状态：**核心闭环已在 2026-08-01 重新验证，原始 Day 1 过程证据不完整**。当前工程可以一键构建并完成 vector add 的 Host → Device → Kernel → Host 正确性闭环；“去掉边界判断”和“非法 launch”两次原始实验没有独立日志，因此不声称原计划全部验收项都已留证。

## 1. 学习目标

从一个 CUDA 工程中理解最小可运行链路：

```text
CPU 准备输入
→ cudaMalloc 分配显存
→ cudaMemcpy H2D
→ kernel<<<grid, block>>>
→ 同步并检查错误
→ cudaMemcpy D2H
→ 与 CPU reference 比较
→ cudaFree 释放显存
```

## 2. 补采环境

| 项目 | 当前验证环境 |
|---|---|
| GPU | NVIDIA GeForce RTX 2080 Ti |
| Compute Capability | 7.5 |
| Driver | 610.62 |
| CUDA Toolkit / nvcc | 13.3 / V13.3.73 |
| CMake | 3.28 及以上 |
| 构建 | Windows、MSVC、Ninja Multi-Config、Release |
| CUDA 架构 | `CMAKE_CUDA_ARCHITECTURES=75` |

这些是补档时的环境，不冒充最初编写 Day 1 代码时的原始记录。

## 3. 核心概念

### Host 与 Device

- **Host**：CPU 和普通系统内存，负责输入准备、CUDA API 调用和结果检查。
- **Device**：GPU 和显存，负责并行执行 kernel。
- Host 指针不能默认当作 Device 指针使用；数据需要按明确方向传输。

### Kernel 启动

```cpp
kernel<<<grid, block>>>(arguments);
```

- `__global__` 表示函数由 Host 发起、在 Device 执行；
- `blockIdx.x` 表示当前 block 编号；
- `threadIdx.x` 表示线程在 block 内的编号；
- `blockDim.x` 表示一个 block 的线程数；
- 全局一维线程索引为 `blockIdx.x * blockDim.x + threadIdx.x`。

### 边界判断

向上取整 grid 后，最后一个 block 可能包含超出 `N` 的线程，因此 kernel 必须保护：

```cpp
if (index < n) {
    c[index] = a[index] + b[index];
}
```

block size 是执行配置，不应成为正确性成立的前提。

## 4. 工程文件映射

| 文件 | 作用 |
|---|---|
| `CMakeLists.txt` | 声明 C++/CUDA 工程、公共库、应用和测试目标 |
| `CMakePresets.json` | 固定 MSVC、Ninja、CUDA 13.3 和 sm_75 配置 |
| `kernels/vector_add.cu` | vector add kernel、启动配置与 Host 封装 |
| `include/vector_add.h` | 对外接口 |
| `apps/vector_add_main.cpp` | CPU reference、尺寸/block 参数化验证和退出码 |
| `include/common/cuda_check.h` | 输出 API、文件、行号和 CUDA 错误文本 |
| `include/common/device_buffer.h` | 后续 Day 3 引入的显存 RAII；当前代码已使用该演进版本 |
| `cmake/Tests.cmake` | 将 `vector_add_app` 注册为 CTest 正确性测试 |

原计划中的 `tests/test_vector_add.cu` 未按该文件名存在；当前由 `vector_add_app` 同时承担独立示例和正确性入口。

## 5. 当前实现流程

`vector_add()` 在 `n <= 0` 时直接返回，非空输入则：

1. 验证 `threads_per_block` 位于 `[1, 1024]`；
2. 分配 `d_a`、`d_b`、`d_c`；
3. 把 A/B 从 Host 拷到 Device；
4. 调用 Device 侧启动函数；
5. 同步以暴露异步执行错误；
6. 把 C 拷回 Host；
7. 由 RAII 自动释放显存。

当前启动函数已演进为 Day 2 的 grid-stride 版本；同文件仍保留最初的单元素索引 kernel，补采验证使用的是当前 grid-stride 路径。

## 6. 构建与验证

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

.\build\windows-ninja\Release\vector_add_app.exe
```

补采覆盖：

```text
N = 0、1、1000、100003
block = 64、128、256
总计 12 组，全部 PASS，程序退出码为 0
```

其中 `100003` 不是 block size 的整数倍，验证了尾部边界；`N=0` 验证了不启动非法空 grid 的早返回路径。

## 7. 错误处理

`CUDA_CHECK(call)` 在 CUDA API 失败时记录：

```text
API 调用文本
源文件
行号
cudaGetErrorString 返回的可读错误
```

`cudaGetLastError()` 用于检查 kernel 启动错误；kernel 执行通常相对 Host 异步，执行期错误需要在同步点暴露。当前仓库的具体错误注入证据归档在 [Day 3](../day03/README.md)。

## 8. 已验证结论与缺口

### 已验证

- Release 工程可以重复 configure/build；
- vector add 对空、单元素、非整齐尺寸和三种 block size 均正确；
- CPU reference 决定 PASS/FAIL，并通过进程退出码接入 CTest；
- 当前错误宏能定位 CUDA API 调用点。

### 原始证据缺口

- 没有保留“删除边界判断后失败、恢复后通过”的独立输出；
- 没有保留 Day 1 当时“非法 launch”的原始日志；
- 原计划的 `tests/test_vector_add.cu` 被应用级验证入口替代。

## 9. 下一步

Day 2 将索引模型扩展为 grid-stride loop 和二维矩阵坐标，重点证明任意尺寸与 block 配置不会改变结果。
