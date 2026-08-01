# Day 3｜显存生命周期与可靠错误处理

> 补档状态：**RAII、API 错误和越界写已在 2026-08-01 重新验证；原计划三类故障注入只保留了两类可复现代码**。错误拷贝大小和 use-after-free 没有独立实验文件，因此不宣称三类全部完成。

## 1. 学习目标

- 用 RAII 管理 `cudaMalloc/cudaFree` 生命周期；
- 禁止显存所有权被意外复制，允许安全移动；
- 区分同步 CUDA API 错误、kernel 启动错误和异步执行错误；
- 使用普通测试与 Compute Sanitizer 定位错误调用和越界地址。

## 2. DeviceBuffer 的所有权模型

`DeviceBuffer<T>` 把“设备指针 + 元素数量”绑定在一个对象中：

```text
构造 → cudaMalloc
正常离开作用域或异常展开 → 析构 → cudaFree
```

核心约束：

| 能力 | 当前实现 | 原因 |
|---|---|---|
| 拷贝构造/赋值 | 删除 | 避免两个对象重复释放同一指针 |
| 移动构造/赋值 | 支持且 `noexcept` | 转移唯一所有权，原对象置空 |
| `data()` | 返回设备指针 | 传给 CUDA API/kernel |
| `size()` | 返回元素数量 | 区分元素数与字节数 |
| 空 buffer | `ptr=nullptr`、不分配 | 支持 `N=0` 早返回路径 |

字节数必须显式计算：

```text
bytes = element_count × sizeof(T)
```

## 3. 错误检查分层

### 同步 API 错误

例如无效的 `cudaMemcpy` 参数通常在 API 调用处立即返回错误，由 `CUDA_CHECK` 打印调用文本、文件、行号和可读错误。

### Kernel 启动错误

`cudaGetLastError()` 检查无效 block/grid、参数等 launch 阶段错误，但不会等待 kernel 执行完成。

### 异步执行错误

越界访问发生在 GPU 执行期间，错误可能到 `cudaDeviceSynchronize()` 或后续同步 API 才暴露。因此 `CUDA_KERNEL_CHECK()` 当前组合了：

```text
cudaGetLastError()
→ cudaDeviceSynchronize()
```

这是适合正确性验证的强检查，不应无条件放进性能计时热路径。

## 4. 测试文件

| 文件 | 验证内容 |
|---|---|
| `include/common/device_buffer.h` | 显存 RAII 与移动语义 |
| `include/common/cuda_check.h` | 统一错误文本与异常抛出 |
| `tests/test_common.cpp` | 编译期验证不可拷贝、可 noexcept 移动；运行期验证 NaN 比较失败 |
| `tests/test_api_error.cu` | 使用空目标地址触发 `cudaMemcpy` invalid argument |
| `tests/test_error_cases.cu` | 故意越界写入，仅在 sanitizer 下可靠定位 |
| `cmake/Tests.cmake` | `WILL_FAIL` 表达“检测到预期错误才算测试通过” |

## 5. 构建与验证

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

& 'D:\software\VS2022\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe' `
    --preset release
```

补采结果：

```text
6/6 tests passed
correctness：4 项
error-handling：2 项
compute-sanitizer：1 项（同时属于 error-handling）
```

CTest 中的错误测试使用 `WILL_FAIL=TRUE`：被测程序成功检测到故意错误并返回非零时，CTest 将其记为 Passed。

详细诊断见 [Compute Sanitizer 与错误注入记录](day03_sanitizer_notes.md)。

## 6. 已验证结论与缺口

### 已验证

- `DeviceBuffer` 不可拷贝、支持 noexcept 移动；
- CUDA API 错误能定位到具体调用、文件和行号；
- Compute Sanitizer 能定位越界 kernel、线程/block 和越界地址；
- 异常展开时，已经成功构造的 RAII 对象会尝试释放资源；
- 正常 Release 测试重复运行没有出现当前可见的资源生命周期错误。

### 结论边界

- 当前没有独立的显存泄漏报告，不能仅凭测试通过宣称“所有路径绝无泄漏”；
- sanitizer 只验证了保留的越界写样例；错误拷贝大小和 use-after-free 没有对应产物；
- GPU 出现 sticky launch failure 后，析构中的 `cudaFree` 也可能收到既有错误；析构函数不能抛异常，当前实现选择忽略释放返回值。

## 7. 下一步

Day 4 使用这些公共组件建立固定随机输入、误差报告、CUDA Event kernel 计时和 Host 端端到端计时。
