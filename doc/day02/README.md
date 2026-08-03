# Day 2：从线程坐标到数据索引

> 本日目标：不背公式，而是理解“GPU 启动了哪些线程、每个线程怎样找到自己负责的数据”。完成后应能独立解释一维线性索引、边界判断、向上取整、grid-stride loop、二维矩阵坐标和行主序地址。

## 1. 开始前只需要知道什么

本章默认读者没有并行编程基础，只需要知道：

- 数组由连续元素组成，下标从 `0` 开始；
- C++ 中 `a[i]` 表示数组 `a` 的第 `i` 个元素；
- 函数可以接收输入、执行计算并返回结果。

不要求事先了解线程、block、grid 或显存。本章会逐个定义。

## 2. 为什么需要“索引映射”

CPU 上最直观的向量加法是：

```cpp
for (int index = 0; index < n; ++index) {
    c[index] = a[index] + b[index];
}
```

这段循环由一个 CPU 线程按顺序处理 `0、1、2……n-1`。CUDA 的做法是启动许多 GPU 线程，让不同线程同时处理不同元素：

```text
线程 0 处理元素 0
线程 1 处理元素 1
线程 2 处理元素 2
……
```

因此最核心的问题不是加法，而是：**一个线程怎样算出自己的全局编号？** 这个线程编号再怎样映射到数组或矩阵下标？

## 3. CUDA 的线程组织层级

CUDA 线程按三层组织：

```text
grid（一次 kernel 启动产生的全部线程）
└── block（线程块）
    └── thread（线程）
```

### 3.1 thread：真正执行 kernel 的个体

每个线程都会执行同一份 kernel 代码，但线程内置坐标不同。`threadIdx.x` 表示当前线程在本 block 的 x 方向编号。

如果一个 block 有 4 个线程：

| 线程 | `threadIdx.x` |
|---:|---:|
| 第 1 个 | 0 |
| 第 2 个 | 1 |
| 第 3 个 | 2 |
| 第 4 个 | 3 |

### 3.2 block：一组可以协作的线程

`blockDim.x` 是每个 block 在 x 方向的线程数。`blockIdx.x` 是当前 block 在 grid 中的编号，同样从 0 开始。

假设启动 3 个 block，每个 block 有 4 个线程：

| block | `blockIdx.x` | 本 block 内的 `threadIdx.x` |
|---:|---:|---|
| 第 1 个 | 0 | 0、1、2、3 |
| 第 2 个 | 1 | 0、1、2、3 |
| 第 3 个 | 2 | 0、1、2、3 |

注意：`threadIdx.x` 会在每个 block 中重新从 0 开始，所以它不是整个 grid 中唯一的编号。

### 3.3 grid：一次 kernel 启动使用的全部 block

`gridDim.x` 是 x 方向的 block 数。启动语法中的前两个参数分别决定 block 数和每个 block 的线程数：

```cpp
kernel<<<blocks, threads_per_block>>>(...);
```

这不是 C++ 模板语法，而是 CUDA 的 kernel 启动语法。

## 4. 一维线性索引是怎样推导出来的

一维全局线程编号为：

```cpp
const int index = blockIdx.x * blockDim.x + threadIdx.x;
```

推导过程是：

1. 当前 block 前面有 `blockIdx.x` 个完整 block；
2. 每个完整 block 有 `blockDim.x` 个线程；
3. 所以前面的线程数是 `blockIdx.x * blockDim.x`；
4. 再加上当前线程在 block 内的编号 `threadIdx.x`。

### 4.1 手算示例

仍然假设每个 block 有 4 个线程：

| `blockIdx.x` | `threadIdx.x` | 计算 | 全局 `index` |
|---:|---:|---|---:|
| 0 | 0 | `0 * 4 + 0` | 0 |
| 0 | 3 | `0 * 4 + 3` | 3 |
| 1 | 0 | `1 * 4 + 0` | 4 |
| 1 | 2 | `1 * 4 + 2` | 6 |
| 2 | 3 | `2 * 4 + 3` | 11 |

因此 3 个 block × 4 个线程会产生全局编号 `0` 到 `11`，没有重复也没有空洞。

## 5. 为什么必须判断 `index < n`

CUDA 一般按完整 block 启动线程，但数组长度不一定正好是 block size 的整数倍。

例如 `n = 10`、每个 block 4 个线程，需要 3 个 block，共启动 12 个线程：

```text
有效数组下标：0～9
启动线程编号：0～11
多出来的线程：10、11
```

所以 kernel 必须写边界保护：

```cpp
if (index < n) {
    c[index] = a[index] + b[index];
}
```

没有这个判断，线程 10 和 11 会访问数组之外的地址，这叫**越界访问**。越界可能产生错误结果、破坏其他数据，也可能直到后面的同步操作才报告错误。

## 6. block 数为什么要向上取整

整数除法会丢掉小数部分：

```text
10 / 4 = 2
```

但 2 个 block 只有 8 个线程，覆盖不了 10 个元素。需要的结果是 3，所以使用向上取整：

```cpp
const int blocks = (n + threads_per_block - 1) / threads_per_block;
```

项目中把它写成不会在 `n > 0` 时溢出加法的等价形式：

```cpp
const int blocks = 1 + (n - 1) / threads_per_block;
```

手算：

```text
n = 10，threads_per_block = 4
blocks = 1 + (10 - 1) / 4
       = 1 + 9 / 4
       = 1 + 2
       = 3
```

### 6.1 `n = 0` 是特殊情况

当数组为空时，不需要启动 kernel。CUDA 不接受 0 个 block 的正常 kernel 启动，因此 Host 封装先返回：

```cpp
if (n <= 0) {
    return;
}
```

## 7. Grid-Stride Loop：线程数少于元素数也能完成任务

“一个线程只处理一个元素”的写法简单，但大型数组可能需要非常多 block。项目中的向量加法把 block 数限制在 4096，然后让线程循环处理后续元素。

```cpp
const int start = blockIdx.x * blockDim.x + threadIdx.x;
const int stride = gridDim.x * blockDim.x;

for (int index = start; index < n; index += stride) {
    c[index] = a[index] + b[index];
}
```

其中：

- `start`：当前线程第一次处理的元素；
- `stride`：整个 grid 的线程总数；
- `index += stride`：跳过其他线程本轮负责的所有元素，进入下一轮。

### 7.1 手算 Grid-Stride Loop

假设总共只启动 6 个线程，却有 14 个元素：

| 线程的 `start` | 负责的元素 |
|---:|---|
| 0 | 0、6、12 |
| 1 | 1、7、13 |
| 2 | 2、8 |
| 3 | 3、9 |
| 4 | 4、10 |
| 5 | 5、11 |

每个有效下标仍然只出现一次。Grid-stride loop 改变的是任务分配方式，不改变向量加法的数学结果。

## 8. 从二维线程坐标映射到矩阵坐标

矩阵天然有“行”和“列”，因此可以使用二维 block 和二维 grid：

```cpp
const dim3 block(16, 16);
```

这表示：

- x 方向 16 个线程，通常负责列；
- y 方向 16 个线程，通常负责行；
- 一个 block 共 `16 * 16 = 256` 个线程。

当前线程负责的全局矩阵坐标为：

```cpp
const int col = blockIdx.x * blockDim.x + threadIdx.x;
const int row = blockIdx.y * blockDim.y + threadIdx.y;
```

记忆方式：

```text
x 是横向移动 → 列 col
y 是纵向移动 → 行 row
```

这只是项目采用的约定，不是 CUDA 强制规定；关键是整个程序保持一致。

## 9. 二维坐标为什么还要转成线性索引

虽然我们把数据理解为矩阵，但 `std::vector<float>` 和 GPU 指针指向的都是一段连续内存。C/C++ 通常按**行主序**保存二维数据：先保存第 0 行，再保存第 1 行。

一个 3 行 4 列的矩阵在内存中是：

```text
矩阵坐标： (0,0) (0,1) (0,2) (0,3) (1,0) (1,1) ...
线性下标：    0     1     2     3     4     5  ...
```

二维坐标转线性下标：

```cpp
const int index = row * width + col;
```

为什么乘 `width`：每跳过一整行，就要跳过一行包含的 `width` 个元素。

### 9.1 手算示例

矩阵宽度 `width = 5`，坐标 `(row = 2, col = 3)`：

```text
前面有 2 个完整行：2 * 5 = 10 个元素
当前行再向右 3 个：10 + 3 = 13
所以 index = 13
```

反向推导也成立：

```text
row = index / width
col = index % width
```

例如 `index = 13`、`width = 5`，得到 `row = 2`、`col = 3`。

## 10. 二维 grid 的大小

宽和高需要分别向上取整：

```cpp
const dim3 grid(
    (width + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);
```

矩阵 kernel 同时保护两个方向：

```cpp
if (row < height && col < width) {
    const int index = row * width + col;
    c[index] = a[index] + b[index];
}
```

例如 17×19 矩阵配 16×16 block，需要 2×2 个 block。实际启动 32×32 个线程，但只有落在 17 行、19 列范围内的线程写结果。

## 11. 代码调用链与数据流

### 11.1 一维向量加法

```text
apps/vector_add_main.cpp
  生成 Host 输入和 CPU 参考答案
        ↓
vector_add(...)
  分配 DeviceBuffer，执行 H2D 拷贝
        ↓
vector_add_device(...)
  计算 launch 配置并启动 grid-stride kernel
        ↓
vector_add_stride_kernel(...)
  每个线程按 start、stride 处理元素
        ↓
  D2H 拷回并与 CPU 结果比较
```

### 11.2 二维矩阵加法

```text
apps/matrix_add_main.cpp
        ↓
matrix_add(...)
  分配显存并复制输入
        ↓
matrix_add_kernel<<<grid, block>>>
  (blockIdx, threadIdx) → (row, col) → linear index
        ↓
  拷回并验证
```

## 12. 本日相关文件

| 文件 | 学习作用 |
|---|---|
| `kernels/vector_add.cu` | 一维全局索引、grid-stride loop、空输入和 launch 参数检查 |
| `apps/vector_add_main.cpp` | 多种长度和 block size 的正确性验证 |
| `kernels/matrix_add.cu` | 二维线程坐标、行主序线性索引、二维边界保护 |
| `apps/matrix_add_main.cpp` | 多种矩阵形状和 CPU reference |
| `include/vector_add.h` | 向量加法的 Host/Device 接口 |
| `include/matrix_add.h` | 矩阵加法接口 |

更细的逐线程推演见 [线性索引与二维映射手算](day02_indexing_walkthrough.md)。

## 13. 构建与运行

在项目根目录执行：

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

.\build\windows-ninja\Release\vector_add_app.exe
.\build\windows-ninja\Release\matrix_add_app.exe
```

## 14. 已验证结果

### 14.1 一维向量

| 输入长度 | block size | 结果 |
|---:|---|---|
| 0 | 64、128、256 | 全部 PASS |
| 1 | 64、128、256 | 全部 PASS |
| 1000 | 64、128、256 | 全部 PASS |
| 100003 | 64、128、256 | 全部 PASS |

### 14.2 二维矩阵

| 高×宽 | 测试意义 | 结果 |
|---:|---|---|
| 0×5 | 空输入 | PASS |
| 1×1 | 最小非空输入 | PASS |
| 3×5 | 小型非整齐形状 | PASS |
| 17×19 | 两个方向都超过 16×16 block 边界 | PASS |
| 100×1000 | 较大非方阵 | PASS |

## 15. 常见误解

### 误解 1：线程编号就是 `threadIdx.x`

`threadIdx.x` 只在当前 block 内唯一。全局编号还必须加上前面 block 占据的线程数。

### 误解 2：启动的线程数必须等于元素数

不需要。多出的线程由边界判断屏蔽；线程较少时也可以用 grid-stride loop 覆盖更多元素。

### 误解 3：矩阵在显存中天然是二维对象

这里的显存是一段线性地址。行和列是程序赋予数据的逻辑含义，最终仍要通过 `row * width + col` 得到线性下标。

### 误解 4：block 越大一定越快

block size 会影响调度和性能，但不应该影响结果。最佳值要通过 Day4 的规范 benchmark 测量，不能只凭线程更多下结论。

### 误解 5：`cudaGetLastError()` 能发现所有越界

它主要检查 kernel 启动阶段错误。GPU 执行期间的越界通常需要同步点或 Compute Sanitizer 才能可靠定位，Day3 会专门讲解。

## 16. 自检题

1. `blockIdx.x = 3`、`blockDim.x = 128`、`threadIdx.x = 7` 时，全局索引是多少？
2. `n = 1000`、每个 block 256 个线程，需要多少个 block？会多启动多少个线程？
3. 总线程数为 8 时，`start = 3` 的线程在 `n = 22` 时处理哪些下标？
4. `width = 7` 时，坐标 `(row = 3, col = 2)` 的线性下标是多少？
5. 线性下标 23 在宽度为 7 的矩阵中对应哪一行、哪一列？

答案：

1. `3 * 128 + 7 = 391`；
2. 4 个 block，共 1024 个线程，多 24 个；
3. 3、11、19；
4. `3 * 7 + 2 = 23`；
5. `row = 23 / 7 = 3`，`col = 23 % 7 = 2`。

## 17. 完成标准

只有满足以下条件，才算完成 Day2：

- 能用自己的话解释 grid、block、thread 三层关系；
- 能从 `blockIdx`、`blockDim`、`threadIdx` 手算全局索引；
- 能解释为什么需要向上取整和边界判断；
- 能手算 grid-stride loop 中某个线程处理的全部下标；
- 能在二维坐标和行主序线性下标之间双向转换；
- 向量与矩阵补采测试全部通过。

下一步 Day3 将回答：显存由谁申请、由谁释放，以及 GPU 异步执行发生错误时，程序为什么经常不能在 kernel 启动那一行立刻报错。
