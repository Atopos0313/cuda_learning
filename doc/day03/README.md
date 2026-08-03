# Day 3：显存生命周期、RAII 与 CUDA 错误定位

> 本日目标：理解“GPU 内存是谁申请、谁释放、错误为什么可能延迟出现”。完成后应能解释 Host/Device 内存、指针与字节数、RAII 所有权、同步和异步错误，以及普通运行与 Compute Sanitizer 各自能发现什么。

## 1. 为什么 Day2 的正确 kernel 还不够

一个 CUDA 程序至少包含三部分：

```text
Host（CPU）准备输入
        ↓ H2D
Device（GPU）执行 kernel
        ↓ D2H
Host 检查或使用结果
```

即使 kernel 的公式完全正确，下面任何问题仍会让程序失败：

- 显存没有成功申请；
- 拷贝方向或字节数写错；
- 已释放的指针仍被使用；
- 两个对象重复释放同一地址；
- kernel 写到分配范围之外；
- 只检查启动错误，没有等待异步执行完成。

所以工程代码需要把资源管理和错误检查变成默认行为，而不是依靠开发者每次都记得补写。

## 2. Host 内存和 Device 内存

### 2.1 Host 内存

普通 `std::vector<float>` 的数据位于 CPU 可以直接访问的内存：

```cpp
std::vector<float> h_input(n);
```

项目使用前缀 `h_` 表示 Host，例如 `h_input`。这只是命名约定，帮助读代码的人识别数据位置。

### 2.2 Device 内存

GPU kernel 使用的输入输出通常位于 Device 内存：

```cpp
float* d_input = nullptr;
cudaMalloc(&d_input, bytes);
```

项目使用前缀 `d_` 表示 Device。`d_input` 这个指针变量本身存放在 CPU 程序里，但它保存的地址指向 GPU 内存。CPU 不能把它当普通数组直接解引用。

### 2.3 元素数不等于字节数

`cudaMalloc` 和 `cudaMemcpy` 的大小参数都是**字节数**：

```cpp
const std::size_t bytes =
    static_cast<std::size_t>(n) * sizeof(float);
```

如果 `n = 1000` 且一个 `float` 为 4 字节，需要 `4000` 字节。把 `n` 直接传给 CUDA API 只会处理 1000 字节，也就是 250 个 `float`。

## 3. 一次完整的显存生命周期

手工管理时，典型流程是：

```text
指针初始化为 nullptr
        ↓
cudaMalloc 申请显存
        ↓
cudaMemcpy H2D 复制输入
        ↓
kernel 使用显存
        ↓
cudaMemcpy D2H 复制结果
        ↓
cudaFree 释放显存
```

这里的“生命周期”是从成功申请到释放之间的时间。生命周期外使用该地址是错误的。

### 3.1 手工管理为什么容易出错

```cpp
float* d_data = nullptr;
CUDA_CHECK(cudaMalloc(&d_data, bytes));

some_operation();  // 如果这里抛出异常

CUDA_CHECK(cudaFree(d_data));
```

如果 `some_operation()` 抛出异常，正常控制流会直接离开当前函数，最后的 `cudaFree` 不会执行，形成资源泄漏。函数有多个提前返回路径时也有同样风险。

## 4. RAII：让对象生命周期管理资源

RAII 是“Resource Acquisition Is Initialization”的缩写，可以理解为：**把资源绑定到 C++ 对象的生命周期**。

项目的 `DeviceBuffer<T>` 在构造时申请显存，在析构时释放显存：

```cpp
{
    DeviceBuffer<float> d_input(n);
    // 在这个作用域中使用 d_input
}  // 离开作用域，析构函数自动执行 cudaFree
```

执行模型：

```text
构造 DeviceBuffer
  └── cudaMalloc

正常 return 或异常离开作用域
  └── C++ 自动调用析构函数
      └── cudaFree
```

RAII 没有让资源“不需要释放”，而是把释放动作放到一个更可靠的自动位置。

## 5. `DeviceBuffer<T>` 逐项理解

### 5.1 模板参数 `T`

`DeviceBuffer<float>` 管理 `float` 数组，`DeviceBuffer<int>` 管理 `int` 数组。申请字节数由 `count * sizeof(T)` 得到。

### 5.2 `explicit` 构造函数

```cpp
explicit DeviceBuffer(std::size_t count);
```

`explicit` 禁止编译器把一个整数悄悄转换成 `DeviceBuffer`。必须明确写：

```cpp
DeviceBuffer<float> buffer(1024);
```

### 5.3 为什么禁止拷贝

如果两个对象同时保存同一个 Device 指针：

```text
对象 A ─┐
        ├── 同一块显存
对象 B ─┘
```

A 和 B 析构时都会调用 `cudaFree`，造成重复释放。因此项目删除拷贝构造和拷贝赋值：

```cpp
DeviceBuffer(const DeviceBuffer&) = delete;
DeviceBuffer& operator=(const DeviceBuffer&) = delete;
```

### 5.4 为什么允许移动

移动不是复制资源，而是转移唯一所有权：

```text
移动前：A → 显存，B → 空
移动后：A → 空，B → 原来的显存
```

实现中使用 `std::exchange` 取走原指针并把原对象置空。置空非常重要，否则原对象析构时仍会释放该地址。

### 5.5 `data()` 和 `size()`

- `data()` 返回 Device 指针，用于 CUDA API 和 kernel；
- `size()` 返回元素数，不是字节数；
- 空 buffer 的 `data()` 为 `nullptr`，`size()` 为 0。

### 5.6 析构函数为什么不抛异常

C++ 异常展开过程中，如果另一个析构函数再次抛出异常，程序可能直接终止。因此析构函数必须保持 `noexcept`，项目会尝试 `cudaFree`，但不会从析构函数抛出释放错误。

这是一种工程取舍：正常 CUDA 调用通过 `CUDA_CHECK` 强检查；析构清理路径保证不制造第二个异常。

## 6. `CUDA_CHECK` 做了什么

CUDA Runtime API 通常返回 `cudaError_t`：

```cpp
const cudaError_t error = cudaMemcpy(...);
```

返回 `cudaSuccess` 表示该 API 调用成功，否则是某种错误码。项目使用：

```cpp
CUDA_CHECK(cudaMemcpy(...));
```

宏会：

1. 保存 API 返回值；
2. 与 `cudaSuccess` 比较；
3. 失败时打印 API 文本、源文件、行号和可读错误消息；
4. 抛出 `std::runtime_error`；
5. 让上层统一捕获，同时触发局部 RAII 对象析构。

错误路径为：

```text
CUDA API 返回失败
  → CUDA_CHECK 抛异常
  → 当前函数停止
  → 已构造的局部对象依次析构
  → DeviceBuffer 尝试释放显存
  → 上层 catch 记录或返回失败
```

## 7. CUDA 错误为什么分层检查

GPU 工作通常相对 CPU **异步**：CPU 发出 kernel 启动命令后，可能在 GPU 真正执行完之前继续向下运行。

因此错误至少分为三类。

### 7.1 同步 API 参数错误

例如向无效目标地址执行 `cudaMemcpy`。API 在返回时通常已经知道参数无效，可以在该调用处立刻由 `CUDA_CHECK` 捕获。

项目样例：`tests/test_api_error.cu`。

### 7.2 Kernel 启动配置错误

例如 block 线程数超过设备限制。启动命令本身无效，可以用：

```cpp
CUDA_CHECK(cudaGetLastError());
```

检查。

### 7.3 Kernel 执行期间错误

例如 kernel 已成功启动，但某个线程在执行时越界写。CPU 调用 `cudaGetLastError()` 时，GPU 可能还没执行到那条错误指令，所以它不能保证发现执行期错误。

需要一个同步点等待 GPU：

```cpp
CUDA_CHECK(cudaDeviceSynchronize());
```

项目提供的强检查组合为：

```cpp
#define CUDA_KERNEL_CHECK()       \
    CUDA_CHECK(cudaGetLastError());       \
    CUDA_CHECK(cudaDeviceSynchronize())
```

它适合教学、测试和排错，不应不加区分地放进性能计时热循环，因为每次设备同步都会阻止 CPU 与 GPU 重叠执行。

## 8. 同步点是什么

同步点可以理解为 CPU 说：“先别继续，等前面指定的 GPU 工作完成。”常见同步点包括：

- `cudaDeviceSynchronize()`：等待设备前面的工作；
- `cudaEventSynchronize(event)`：等待某个 Event；
- 阻塞式 D2H `cudaMemcpy`：CPU 要拿到结果，通常会等相关 GPU 工作完成。

注意：同步可以帮助暴露异步错误，但同步越多不代表程序越正确或越高效。应该在正确的边界同步。

## 9. 为什么普通运行可能漏掉越界

显存分配和硬件访问以一定粒度工作。某个线程只越过逻辑数组末尾几个字节时，地址有时仍落在底层可映射区域内：

- 程序可能没有立刻崩溃；
- `cudaDeviceSynchronize()` 也不一定报告；
- 但这个访问在程序语义上仍然是错误的。

所以不能用“程序没报错”证明不存在越界。Compute Sanitizer 会跟踪分配边界，能更可靠地识别这种访问。

## 10. Compute Sanitizer 的作用

本项目使用 `memcheck` 工具运行故意越界的程序：

```powershell
compute-sanitizer --tool memcheck `
    .\build\windows-ninja\Release\test_error_cases.exe
```

报告会指出：

- 错误类型，例如 Invalid global write；
- 出错 kernel 和源代码位置；
- 哪个 thread、哪个 block 执行了错误访问；
- 访问地址相对合法分配范围的位置。

Compute Sanitizer 是诊断工具，不是结果验证的替代品。正确的工程流程同时需要 CPU reference、边界测试、API 检查和 sanitizer。

## 11. CTest 中“预期失败”为何显示 Passed

`test_api_error` 和 sanitizer 样例故意制造错误。它们正确工作的标志就是程序返回非零状态。

CMake 使用：

```cmake
set_tests_properties(test_name PROPERTIES WILL_FAIL TRUE)
```

告诉 CTest：“这个程序失败才符合预期。”因此测试显示 Passed 的含义是**预期错误被成功触发和检测**，不是被测操作正常完成。

## 12. 本日相关文件

| 文件 | 学习作用 |
|---|---|
| `include/common/device_buffer.h` | Device 显存 RAII、唯一所有权和移动语义 |
| `include/common/cuda_check.h` | 统一检查 CUDA API 和 kernel 错误 |
| `tests/test_common.cpp` | 类型特征、移动能力和比较器异常值测试 |
| `tests/test_api_error.cu` | 可立即返回的 CUDA API 参数错误 |
| `tests/test_error_cases.cu` | 故意执行越界写，供 sanitizer 定位 |
| `cmake/Tests.cmake` | 正确性测试、预期失败和 sanitizer 集成 |

实测诊断输出见 [Compute Sanitizer 与错误注入记录](day03_sanitizer_notes.md)。

## 13. 构建与验证

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

& 'D:\software\VS2022\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe' `
    --preset release --output-on-failure
```

补采结果：

```text
6/6 tests passed
```

其中包括向量/矩阵/内存访问正确性、公共组件、预期 API 错误，以及安装 Compute Sanitizer 时的预期越界错误。

## 14. 常见误解

### 误解 1：用了 RAII 就绝不会发生显存错误

RAII 主要解决生命周期和释放路径问题。它不能自动阻止 kernel 越界、字节数错误或错误的并发访问。

### 误解 2：指针变量叫 `d_input`，它本身就在 GPU 上

指针变量由 Host 代码持有；它保存的地址属于 Device 地址空间。命名只在提醒地址指向哪里。

### 误解 3：`cudaGetLastError()` 成功就说明 kernel 计算正确

它只能说明当前可见的 CUDA 错误状态没有报告启动错误。数学结果仍要与 reference 比较，执行期错误仍可能要到同步点出现。

### 误解 4：同步越多越安全

同步能明确完成边界并帮助定位错误，但过度同步会破坏异步执行和流水重叠。调试检查与性能路径要区分。

### 误解 5：测试显示 Passed，故意越界程序就没有错误

`WILL_FAIL` 测试的 Passed 表示错误被按预期发现。必须阅读测试设计，不能只看一个单词。

## 15. 自检题

1. `DeviceBuffer<float>(1000)` 通常申请多少字节？
2. 为什么 `DeviceBuffer` 禁止拷贝但允许移动？
3. `cudaGetLastError()` 与 `cudaDeviceSynchronize()` 分别主要暴露哪一阶段的错误？
4. 为什么析构函数不应该抛异常？
5. 普通运行没有报错，为什么仍不能证明没有越界？

## 16. 结论边界

当前证据可以确认：

- `DeviceBuffer` 不可拷贝、可 `noexcept` 移动；
- API 错误能定位到具体调用、文件和行号；
- Compute Sanitizer 能定位保留的越界写样例；
- Release 配置下当前 6 项 CTest 全部通过。

当前证据不能声称：

- 所有可能的资源泄漏都已被证明不存在；
- 错误拷贝大小和 use-after-free 已有独立故障注入样例；
- 通过一次 sanitizer 就能证明任意输入和任意并发路径都安全。

## 17. 完成标准

- 能画出 H2D、kernel、D2H 的数据路径；
- 能区分元素数与字节数；
- 能解释 RAII、禁止拷贝和移动后置空；
- 能区分 API 错误、launch 错误和执行期错误；
- 能解释为什么 sanitizer 的预期失败在 CTest 中显示 Passed；
- 当前测试全部通过。

下一步 Day4 将建立可信的 benchmark：先证明结果正确，再讨论计时范围、预热、重复采样、中位数和有效带宽。
