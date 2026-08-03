# Day 3 补充：Compute Sanitizer 与错误注入记录

## 1. 记录目的

本记录保存 2026-08-01 的实际错误证据，并说明怎样阅读诊断信息。错误样例是故意编写的，不应复制到正常业务 kernel 中。

## 2. 同步 API 错误

样例：`tests/test_api_error.cu`。

程序把空目标地址传给 `cudaMemcpy`，`CUDA_CHECK` 捕获到：

```text
API: cudaMemcpy(...)
File: tests/test_api_error.cu
Line: 20
Error: invalid argument
```

这说明错误在 CUDA API 返回时已经可见，不需要等待 kernel。

## 3. 故意越界的 kernel

样例：`tests/test_error_cases.cu`。

程序只申请一个 `float`，kernel 却写入分配范围之外。普通 Runtime 执行并不保证可靠报告如此小的越界，所以 CTest 只在找到 `compute-sanitizer` 时添加该测试。

实测 `memcheck` 报告的关键信息：

```text
Invalid __global__ write
tests/test_error_cases.cu:14
thread (0,0,0), block (0,0,0)
Address is 4093 bytes after a 4-byte allocation
```

## 4. 每一项信息怎样理解

| 报告字段 | 含义 | 排查动作 |
|---|---|---|
| Invalid global write | kernel 向不属于该分配的全局内存写入 | 检查目标指针、索引和边界条件 |
| 文件与行号 | 触发非法访问的源码位置 | 回到 kernel 对应行，不要只看后面的同步 API |
| thread / block | 哪个逻辑线程执行了错误访问 | 代入索引公式手算该线程地址 |
| allocation relation | 地址位于合法分配之前或之后多远 | 检查申请字节数和索引单位 |

异步错误有时会在 `cudaDeviceSynchronize` 或 D2H 拷贝处被 Runtime 报出，但真正错误往往发生在更早的 kernel 指令。Sanitizer 的源码位置比“最后看到错误的同步调用”更接近根因。

## 5. 推荐排查顺序

1. 确认分配的元素数和字节数；
2. 手算报告中 thread/block 的全局索引；
3. 检查所有维度的边界判断；
4. 检查指针是否已经释放或指向错误缓冲区；
5. 检查 H2D/D2H 方向和拷贝大小；
6. 修复后重新运行正常测试与 sanitizer。

## 6. 运行方式

单独运行：

```powershell
compute-sanitizer --tool memcheck --error-exitcode 1 `
    .\build\windows-ninja\Release\test_error_cases.exe
```

通过 CTest 运行全部错误处理测试：

```powershell
ctest --preset release -L error-handling --output-on-failure
```

## 7. 不能从本次记录推出什么

- 没有报告不等于所有路径都无越界；测试覆盖范围仍由输入决定；
- `memcheck` 主要检查内存问题，不替代竞争检查、结果比较或性能分析；
- 当前仓库只保留 API 参数错误和越界写两类故障注入，不能声称已覆盖所有 CUDA 错误类型。
