# Day 2｜索引、Grid-Stride Loop 与任意形状

> 补档状态：**核心 1D/2D 索引目标已在 2026-08-01 验证，原计划产物存在结构差异**。当前仓库用 `vector_add.cu` 和 `matrix_add.cu` 分别承载一维、二维练习；没有独立的 `elementwise.cu`、`launch_config.h` 或 `test_indexing.cu`。

## 1. 学习目标

- 从线程坐标推导一维数据索引；
- 用 grid-stride loop 覆盖超过固定 grid 容量的数据；
- 从二维 block/grid 坐标推导矩阵 `(row, col)`；
- 正确处理空输入、质数/非整齐尺寸和不同 block size。

## 2. 一维索引与 Grid-Stride Loop

普通单元素索引为：

```text
start = blockIdx.x × blockDim.x + threadIdx.x
```

Grid-stride loop 再计算整个 grid 的线程跨度：

```text
stride = gridDim.x × blockDim.x
```

每个线程处理：

```cpp
for (int index = start; index < n; index += stride) {
    c[index] = a[index] + b[index];
}
```

当前实现将 grid 上限限制为 4096 blocks。数据超过一次覆盖范围时，已启动线程继续按 `stride` 处理后续元素，避免 grid 大小无限增长。

### 最小规则

- `start` 决定线程第一次处理哪个元素；
- `stride` 是所有已启动线程的总数；
- 循环条件仍然必须检查 `index < n`；
- block size 影响线程组织和性能，但不改变数学结果。

## 3. 二维矩阵索引

矩阵使用行主序存储。线程的全局坐标为：

```text
col = blockIdx.x × blockDim.x + threadIdx.x
row = blockIdx.y × blockDim.y + threadIdx.y
```

二维坐标转一维地址：

```text
index = row × width + col
```

边界保护：

```cpp
if (col < width && row < height) {
    const int index = row * width + col;
    c[index] = a[index] + b[index];
}
```

当前 matrix add 使用 `dim3 block(16, 16)`，grid 在行列两个方向分别向上取整。

## 4. 代码文件

| 文件 | 作用 |
|---|---|
| `kernels/vector_add.cu` | grid-stride loop、一维早返回和 block 参数验证 |
| `apps/vector_add_main.cpp` | 4 个长度 × 3 个 block size 的验证 |
| `kernels/matrix_add.cu` | 二维 grid/block、行主序索引和边界判断 |
| `apps/matrix_add_main.cpp` | 5 个二维形状的 CPU reference 验证 |
| `include/vector_add.h` | 一维 Host/Device 接口 |
| `include/matrix_add.h` | 二维 Host 接口 |

当前 `ceil_div` 逻辑以内联表达式存在：

```cpp
(width + block.x - 1) / block.x
```

尚未抽取成原计划要求的通用 `launch_config.h`。

## 5. 构建与验证

```powershell
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass `
    -File .\scripts\build.ps1 -Configuration Release

.\build\windows-ninja\Release\vector_add_app.exe
.\build\windows-ninja\Release\matrix_add_app.exe
```

### 一维补采结果

| 输入长度 | block size | 结果 |
|---:|---|---|
| 0 | 64、128、256 | 全部 PASS |
| 1 | 64、128、256 | 全部 PASS |
| 1000 | 64、128、256 | 全部 PASS |
| 100003 | 64、128、256 | 全部 PASS |

### 二维补采结果

| height × width | 特征 | 结果 |
|---:|---|---|
| 0 × 5 | 空输入 | PASS |
| 1 × 1 | 单元素 | PASS |
| 3 × 5 | 小型非整齐形状 | PASS |
| 17 × 19 | 两个维度均越过 16×16 block 边界 | PASS |
| 100 × 1000 | 较大矩阵 | PASS |

## 6. 为什么 `N=0` 不能照常 launch

当 `n=0` 时，按向上取整公式可能得到 0 blocks。CUDA kernel 不能使用非法的 0 维 grid，因此 Host 封装先返回。二维版本同样在 `height <= 0 || width <= 0` 时返回。

## 7. 已验证结论与缺口

### 已验证

- grid-stride vector add 在三种 block size 下结果一致；
- 1D/2D 都正确处理空输入和非整齐尾部；
- 二维线程坐标能映射到行主序线性地址；
- block size 不参与数学定义。

### 与原计划的差距

- 1D 当前只有 4 个尺寸、2D 当前只有 5 个形状，未达到“每类至少 8 组”的字面要求；
- 没有独立通用的 `ceil_div`/launch 配置头文件；
- 没有保留扁平 1D grid 与二维 grid 两种 matrix kernel 的 A/B 对照；
- 代码产物名称与计划不同，但核心索引能力已由现有文件覆盖。

## 8. 下一步

Day 3 将裸设备内存和错误检查封装为可复用基础设施，重点处理所有权、移动语义和异步错误暴露。
