# 教学代码规范

## 1. 目标

本仓库首先是零基础学习工程，其次才是算子集合。代码应同时满足：

- 能编译、能测试、结果正确；
- 命名能说明数据位置和职责；
- 格式一致，减少与知识点无关的阅读负担；
- 注释准确解释机制，不保留大段失效代码；
- 优化版本保留清晰基线，便于公平比较。

`.editorconfig` 统一 UTF-8、LF、4 空格缩进和文件末尾换行。

## 2. 文件职责

| 目录 | 职责 |
|---|---|
| `include/` | 稳定接口和公共组件 |
| `kernels/` | kernel、紧邻 kernel 的校验和 Device launch 封装 |
| `apps/` | 组织完整可运行流程和面向学习者的结果输出 |
| `tests/` | 正确性、边界和预期错误验证 |
| `benchmarks/` | 性能测量，不承担首次正确性发现 |
| `doc/` | 概念、推导、运行证据、结论和限制 |

## 3. 命名

- Host 指针或容器使用 `h_` 前缀，例如 `h_input`；
- Device 指针使用 `d_` 前缀，例如 `d_input`；
- 元素数量使用 `n` 或 `element_count`，字节数量使用 `bytes`；
- 每 block 线程数统一使用 `threads_per_block`；
- 常量使用 `kName`，例如 `kWarpSize`；
- 布尔变量使用可读状态，例如 `passed`、`all_passed`；
- 对外 Device 启动接口使用 `_device` 后缀；
- kernel 使用 `_kernel` 后缀。

不要用 `N`、`n`、`size`、`count` 混指同一层含义。必须区分元素数、向量数、block 数和字节数。

## 4. Kernel 结构

简单逐元素 kernel 推荐顺序：

```cpp
__global__ void operation_kernel(...)
{
    const int index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (index < n) {
        // 只执行该线程负责的数据操作
    }
}
```

二维 kernel 明确区分 `row`、`col` 和线性 `index`。复杂映射在每日文档中提供小规模手算表，源码只保留能够帮助理解机制的短注释。

## 5. Host/Device 接口

Device 接口负责：

- 处理空输入；
- 校验 launch 参数；
- 计算 grid/block；
- 在指定 stream 启动 kernel；
- 使用 `cudaGetLastError()` 检查启动错误。

完整 Host 包装可以负责：

- 分配 `DeviceBuffer`；
- H2D/D2H 拷贝；
- 在正确性边界同步；
- 把 Device 接口组合成可直接运行的操作。

性能循环中不默认加入 `cudaDeviceSynchronize()`；完成边界由 Event 或数据依赖建立。

## 6. 注释原则

好的注释解释：

- 公式为什么成立；
- 特殊分支解决什么边界；
- 某个同步或对齐检查为何必要；
- 当前实现的限制和假设。

应删除：

- 已经由版本控制保存的大段旧实现；
- 与代码不一致的历史说明；
- 逐字翻译明显语句但不提供机制的注释；
- 把架构相关经验写成 CUDA 永久规则的断言。

## 7. 数值与性能证据

- 所有 benchmark 先执行 reference 验证；
- 浮点结果使用明确 `atol`/`rtol`；
- 报告 Release 配置、GPU、CUDA、规模、block、预热和迭代数；
- kernel-only 和 end-to-end 必须分别命名；
- CSV 保存原始机器可读结果；
- 没有 profiler 计数器时，用“与机制一致”而不是“已证明硬件根因”。

## 8. 修改检查单

- [ ] 没有改变不在任务范围内的学习阶段；
- [ ] 新接口命名和现有约定一致；
- [ ] 空输入、非整齐尾部和非法 launch 有处理；
- [ ] 正确性测试通过；
- [ ] 性能修改使用 Release benchmark；
- [ ] 文档链接和源码路径有效；
- [ ] 没有把生成目录或临时 profiler 文件加入版本控制。
