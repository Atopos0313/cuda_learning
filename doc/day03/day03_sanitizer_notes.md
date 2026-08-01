# Day 3 专题｜CUDA API 错误与 Compute Sanitizer 诊断

> 本文记录 2026-08-01 对当前 Day 3 错误样例的补采验证。它证明现有两类故障可以被定位，不代表原计划中所有错误类型都已保存。

## 1. 普通 CUDA API 参数错误

`tests/test_api_error.cu` 将 `cudaMemcpy` 的目标指针故意设为 `nullptr`。实际输出的核心部分：

```text
CUDA Error
API: cudaMemcpy(nullptr, buffer.data(), sizeof(float), cudaMemcpyDeviceToHost)
File: ...\tests\test_api_error.cu
Line: 20
Error: invalid argument
Caught exception: invalid argument
```

这证明错误宏能保留调用表达式和源码位置，而不是只输出“程序失败”。

## 2. Kernel 越界写

测试分配 1 个 float，却让线程写入：

```cpp
data[index + 1024] = 1.0F;
```

普通 CUDA Runtime 不保证仅凭分配边界立即发现这种访问，因为底层分配粒度可能大于请求大小。因此 CTest 在检测到 Compute Sanitizer 时使用：

```powershell
compute-sanitizer --tool memcheck --error-exitcode 1 test_error_cases.exe
```

补采报告的关键证据：

```text
Invalid __global__ write of size 4 bytes
at out_of_bounds_kernel(float *) ... test_error_cases.cu:14
by thread (0,0,0) in block (0,0,0)
Access ... is 4,093 bytes after the nearest allocation ... of size 4 bytes
Program hit cudaErrorLaunchFailure ... on cudaDeviceSynchronize
```

## 3. 为什么错误在同步点出现

Kernel launch 对 Host 通常是异步的：

```text
CPU 提交 kernel
→ launch API 返回
→ GPU 稍后执行并发生越界
→ cudaDeviceSynchronize 等待
→ 执行错误在同步点返回
```

所以 `cudaGetLastError()` 与同步检查解决的是不同阶段的问题。

## 4. CTest 为什么显示 Passed

两个错误样例都应该返回非零，`cmake/Tests.cmake` 对它们设置了 `WILL_FAIL TRUE`。因此：

```text
错误成功暴露 → 程序非零退出 → CTest 判定 Passed
错误没有暴露 → 程序零退出 → CTest 反而判定 Failed
```

## 5. 尚未补齐的故障实验

- 错误 memcpy 字节数；
- use-after-free；
- 独立 leak-check 报告。

这些内容只有在新增可复现代码和实际诊断输出后，才能标记为已验证。
