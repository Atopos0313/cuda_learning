# Day 9 C++ / CUDA 代码阅读问答

本文整理 Day 9 编写 stream pipeline 时实际遇到的 C++ 与 CUDA 接口问题。它们不是脱离工程的语法清单，而是帮助读懂本项目 RAII buffer、stream、event 和 benchmark 代码的补充材料。

## 1. 为什么异步 H2D / D2H 通常要配 pinned memory

更准确的说法是：**涉及 CPU 内存的异步 H2D/D2H 拷贝，若要可靠地由 GPU DMA 引擎在后台传输，Host 端通常必须使用 pinned memory。** kernel 的异步启动和纯 Device-to-Device 拷贝不依赖 Host pinned memory。

普通 `std::vector` 使用 pageable memory。操作系统可以换出或重新映射这些内存页，因此 GPU DMA 引擎不能把它当作位置始终稳定的后台传输源或目标。当 `cudaMemcpyAsync()` 收到 pageable 指针时，Runtime 可能需要先执行 staging：

```text
pageable Host memory
    -> Runtime/CPU 拷入内部 pinned 暂存区
    -> GPU DMA
    -> Device memory
```

前面的 CPU 拷贝可能使 API 调用发生阻塞，所以无法可靠获得“CPU 很快返回、DMA 在后台继续”的行为。

`cudaMallocHost()` 分配的 pinned memory 会锁定对应物理页，使 DMA 引擎能够稳定访问：

```text
pinned Host memory -> GPU DMA -> Device memory
```

但 pinned memory 只是可靠异步传输和潜在 copy/compute overlap 的必要条件之一，不是充分条件：

- 它不保证 `cudaMemcpyAsync()` 永远立即返回；
- 它不保证 copy engine 和 compute engine 一定重叠；
- H2D 提交后，在对应 Event/Stream 完成前，CPU 不能改写输入 pinned buffer；
- D2H 提交后，在对应 Event/Stream 完成前，CPU 不能读取或释放输出 pinned buffer；
- pinned memory 会占用有限的系统锁页资源，应尽量一次分配、循环复用。

Day 9 的 Nsight Systems 结果正好说明这个边界：输入输出都使用了 pinned memory，也使用了 4 条 stream，但时间线上仍然没有 copy/compute overlap。

## 2. `const` 与 `constexpr`

`const` 保证对象初始化后不能通过该名字修改；它的值可以到运行时才确定：

```cpp
const int n = get_size();
```

`constexpr` 要求初始化表达式能够在编译期求值，并且对象本身也是不可修改的：

```cpp
constexpr int block_size = 256;
constexpr std::size_t chunk_size = 1'048'576;
```

对变量而言，可以记为：

```text
constexpr => const
const      !=> constexpr
```

`std::array<T, N>` 的 `N` 是类型的一部分，必须在编译期确定，所以固定 stream 数适合使用 `constexpr`。如果 stream 数来自命令行，则应使用运行时变量和 `std::vector`。

## 3. `DeviceBuffer<float>(chunk_size)` 为什么没有变量名

在下面的初始化中：

```cpp
std::array<DeviceBuffer<float>, 2> chunk_inputs{
    DeviceBuffer<float>(chunk_size),
    DeviceBuffer<float>(chunk_size)
};
```

两个 `DeviceBuffer<float>(chunk_size)` 都是在初始化位置构造的临时对象，真正长期存在并可寻址的对象是：

```cpp
chunk_inputs[0]
chunk_inputs[1]
```

这与先分别声明两个 buffer 的分配效果相同，但数组形式可以用 `stream_id` 统一索引：

```cpp
chunk_inputs[stream_id];
chunk_outputs[stream_id];
streams[stream_id];
```

如果 `DeviceBuffer<float>(chunk_size);` 单独占一条语句，它会在语句结束时立即析构，通常没有用途。构造函数声明为 `explicit` 后，也不能把普通整数隐式转换成 `DeviceBuffer`，因此初始化时明确写出类型更安全。

## 4. `cudaMalloc(&ptr_, ...)` 为什么不需要 `reinterpret_cast<void**>`

底层 C 风格接口接收 `void**`。标准 C++ 不允许把 `T**` 隐式转换为 `void**`，因为那会破坏类型安全，所以传统写法常见：

```cpp
CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), bytes));
```

但 CUDA 的 C++ 头文件还提供了类型化模板重载，因此在本项目的 `.cu` / CUDA C++ 编译环境中可以直接写：

```cpp
CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
```

这不是 C++ 自动把 `T**` 隐式转成了 `void**`，而是编译器选择了 CUDA 提供的 C++ 重载。直接写更简洁；如果代码面对旧工具链或纯 C 接口，显式转换仍可能需要。

## 5. `(void)cudaFreeHost(ptr_)` 前面的 `void`

`cudaFreeHost()` 返回 `cudaError_t`。析构函数通常声明为 `noexcept`，不能安全地通过异常报告清理失败，因此这里显式丢弃返回值：

```cpp
(void)cudaFreeHost(ptr_);
```

它表达的是“我知道这个函数有返回值，并有意忽略”，不是把 `ptr_` 转换成 `void*`。正常业务路径中的 CUDA 调用仍应使用 `CUDA_CHECK`；析构清理是特殊边界。

## 6. 两个 `data()` 有什么区别

```cpp
T* data() noexcept;
const T* data() const noexcept;
```

- 非 `const` 对象调用第一个版本，返回 `T*`，允许修改元素；
- `const` 对象调用第二个版本，返回 `const T*`，只允许读取元素；
- 第二个函数末尾的 `const` 表示该成员函数不会修改对象的可观察成员状态；
- 返回类型中的 `const` 表示不能通过返回指针修改所指元素。

这是 const-correctness：对象是否只读会自然传播到它提供的数据视图。

## 7. `float* const` 与 `const float*`

二者不是一个意思：

```cpp
float* const p = buffer;  // 指针本身不能改，所指 float 可以改
const float* p = buffer;  // 指针可以改指别处，所指 float 不能通过 p 修改
```

还可以组合为：

```cpp
const float* const p = buffer; // 指针和所指数据都不能改
```

阅读时从变量名向外结合：`p` 先遇到右侧 `const`，就是“p 是 const 指针”。

## 8. `void synchronize() const` 如何阅读

```cpp
void synchronize() const
{
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}
```

- 第一个 `void`：没有返回值；
- `synchronize()`：不接收参数；
- 末尾 `const`：不会修改包装对象内部保存的 `stream_` handle；
- 它仍然可以等待外部 CUDA stream 的进度，因为外部 GPU 状态变化不等同于修改这个 C++ 包装对象的成员。

## 9. `CudaCompletionEvent` 各成员的职责

该类用 RAII 管理一个 `cudaEvent_t`：

- 构造函数：以 `cudaEventDisableTiming` 创建只负责完成同步的 Event；
- 删除拷贝构造/拷贝赋值：避免两个对象同时认为自己拥有同一个 handle，最终重复销毁；
- 移动构造：使用 `std::exchange` 把 handle 所有权转给新对象，并把旧对象置空；
- 删除移动赋值：当前项目不需要在已存在对象之间重新转移所有权，缩小状态空间；
- 析构函数：对象离开作用域时销毁 Event；
- `record(stream)`：把 Event 记录在指定 stream 当前队尾；
- `synchronize()`：让 Host 等到该 Event 之前的工作全部完成；
- `get()`：只暴露原始 handle，供必须接收 `cudaEvent_t` 的 CUDA API 使用。

`get()` 不是 CUDA 语法要求；它是 RAII 包装器与原始 CUDA C API 之间的桥。也可以命名为 `native_handle()`。如果项目从不需要在类外传递原始 handle，可以不提供它。

## 10. `reserve()` 与 `emplace_back()`

```cpp
streams.reserve(static_cast<std::size_t>(stream_count));
streams.emplace_back();
```

`reserve(n)` 只预留至少能容纳 `n` 个元素的容量，不会创建元素，`size()` 仍为 0。这样随后追加多个 RAII 对象时，可以避免 vector 扩容以及已有对象的移动。

`emplace_back(args...)` 是 `std::vector` 的成员函数，它直接在 vector 尾部用这些参数构造一个新元素。例如：

```cpp
chunk_inputs.emplace_back(chunk_size);
```

等价意图是“在末尾构造一个 `DeviceBuffer<float>(chunk_size)`”。`static_cast<std::size_t>` 用来把已经验证为非负的 stream 数转换成容器使用的无符号大小类型；若值来自用户输入，应先检查它大于 0。

## 11. `std::fixed`、`std::setprecision()` 与 `std::scientific`

它们控制浮点数输出格式，并会持续影响后续输出，直到再次修改：

```cpp
std::cout << std::fixed << std::setprecision(3) << 12.34567;
// 12.346

std::cout << std::scientific << std::setprecision(3) << 12.34567;
// 1.235e+01
```

在 `fixed` / `scientific` 模式下，`setprecision(3)` 表示小数点后 3 位。Day 9 CSV 用固定小数显示毫秒和加速比，用科学计数法显示误差，便于机器读取和人工比较。

## 12. Nsight Systems 中容易混淆的字段

- CUDA API 的 `Duration`：CPU 线程调用 `cudaMemcpyAsync()` 或 `cudaLaunchKernel()` 并向 Runtime 提交任务所花的时间，不是 GPU 真正执行时间；
- GPU activity 的 `Duration`：Memcpy 或 kernel 在 GPU 时间线上的实际持续时间；
- `Latency`：CPU 提交活动后，到 GPU 真正开始活动之间的等待时间；
- `cudaProfilerStart` 的长耗时：profiler 启动/初始化成本，不属于被分析的 H2D-kernel-D2H 工作负载；
- `cudaEventSynchronize` 共 4 次：对应 4 条活动 stream 各自的尾部 completion event。第一次等待可能承担主要等待时间，后续 Event 往往已经接近完成。

普通 benchmark 路径执行 20 轮预热和 50 轮正式测量；`--profile` 是独立的专用路径：采集范围外预热 1 轮，`cudaProfilerStart()` 与 `cudaProfilerStop()` 之间只执行 1 轮。Nsight Systems 不是从 50 轮中随机挑一轮，而是程序主动只把这 1 轮放进 capture range。报告中的 17 次 H2D、17 次 kernel、17 次 D2H，正好对应 `ceil(16,777,219 / 1,048,576) = 17` 个 chunk。

## 13. 本轮形成的代码阅读方法

阅读 CUDA RAII 和异步代码时，可以按以下顺序检查：

1. 谁拥有资源，谁负责销毁；
2. copy、kernel、Event 分别被提交到哪条 stream；
3. 同一 buffer 何时可以复用；
4. Host 在哪里等待，等待的是单条 stream、单个 Event，还是整个 Device；
5. 指针的 `const` 修饰的是指针还是数据；
6. Nsight 中看到的是 CPU API 时间还是 GPU activity 时间。

这套顺序能同时发现重复释放、过早读取、buffer data race、隐式同步和指标误读。
