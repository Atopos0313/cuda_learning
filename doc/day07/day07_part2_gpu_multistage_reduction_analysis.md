# Day 07 Part 2：全 GPU 多轮归约与 Ping-Pong Buffer

## 1. 问题与目标

第一阶段的 GPU 只把 N 个输入变成多个 block 部分和，最终仍由 CPU 合并。这样
虽然正确，但会复制多个中间结果回 CPU，并让 CPU 参与本应继续并行缩减的工作。

本阶段目标是：

```text
N 个输入
→ GPU 第一轮部分和
→ GPU 第二轮部分和
→ ...
→ GPU 上只剩 1 个结果
→ D2H 复制 1 个 float
```

## 2. 每轮数据量

当前一个线程加载一个元素，一个 block 输出一个结果，因此：

```text
next_count = ceil(current_count / threads_per_block)
```

代码使用：

```cpp
reduction_block_count(
    current_count,
    threads_per_block
);
```

例如 `N=100003, block=256`：

```text
100003 → 391 → 2 → 1
```

需要三轮 kernel。

## 3. 为什么可以重复调用同一个 kernel

一轮 kernel 只关心：

- 输入指针；
- 输出指针；
- 当前输入数量；
- block 大小。

第一轮输入是原始数据，后续输入是上一轮的部分结果。对 kernel 而言，它们都是
连续的 `float` 数组，所以不需要为第二轮编写另一套算法。

## 4. 为什么不能安全原地归约

如果让输入和输出指向同一缓冲区，一个较快的 block 可能先写入 `buffer[1]`，
而另一个 block 尚未把旧的 `buffer[1]` 搬入 shared memory。普通 kernel 内没有
跨 block barrier，因此可能发生跨 block 读写竞争。

解决方法是两块 workspace：

```text
第一轮：d_input → d_ping
第二轮：d_ping  → d_pong
第三轮：d_pong  → d_ping
```

`std::swap(current_output, alternate_output)` 只交换 host 端两个指针保存的地址，
不会复制 GPU 缓冲区内容。

## 5. 核心控制流程

```cpp
const float* current_input = d_input.data();
float* current_output = d_ping.data();
float* alternate_output = d_pong.data();
int current_count = n;

while (current_count > 1) {
    const int next_count =
        reduction_block_count(
            current_count,
            threads_per_block
        );

    reduce_sum_blocks_device(
        current_input,
        current_output,
        current_count,
        threads_per_block
    );

    current_input = current_output;
    current_count = next_count;

    std::swap(
        current_output,
        alternate_output
    );
}
```

最后 `current_input` 指向唯一结果，只复制一个 `float` 回 host。

## 6. Kernel 之间为什么不调用 `cudaDeviceSynchronize`

所有 kernel 都提交到同一个 stream。同一 stream 保证：

```text
kernel 1 完成
→ kernel 2 才能执行
→ kernel 3 才能执行
```

因此后续 kernel 不会提前读取未完成的部分和。这里的阶段边界来自 kernel launch
顺序，不是 `__syncthreads()`。

两者区别：

| 机制 | 同步范围 |
|---|---|
| `__syncthreads()` | 一个 block |
| 同 stream 的 kernel 顺序 | 不同 kernel 阶段 |
| `cudaDeviceSynchronize()` | host 等待整个 device |

多轮之间不需要让 host 每轮等待 device。

## 7. 正确性结果

测试覆盖：

- `N=0、1、31、32、33、100003`；
- block `32、64、128、256`；
- random 与纯负数；
- 极端浮点输入。

普通用例全部 `PASS`。`N=100003` 的轮数：

| block | 变化过程 | rounds |
|---:|---|---:|
| 32 | 100003→3126→98→4→1 | 4 |
| 64 | 100003→1563→25→1 | 3 |
| 128 | 100003→782→7→1 | 3 |
| 256 | 100003→391→2→1 | 3 |

## 8. 浮点误差

第一阶段版本由 CPU 使用 double 合并部分和；全 GPU 版本继续使用 float。两者
加法顺序与中间精度不同，因此结果不要求逐位相等。

正确性容限：

```text
1e-4 + 1e-6 × |cpu_sum|
```

极端数据：

```text
[1e20, -1e20, 1, -2]
```

CPU 顺序得到 `-1`，GPU 树形顺序得到 `0`。这是浮点加法不满足严格结合律的
观察用例，记录为 `OBSERVE`，不作为 kernel 错误。

## 9. 结论与局限

**已验证：**

- ping-pong 指针更新顺序正确；
- 多轮结果保持在 GPU；
- 同 stream 顺序足以表达轮次依赖；
- 空输入和尾部数据处理正确。

**局限：**

- host 仍需逐轮提交 kernel；
- buffer 由调用者预先分配；
- 本阶段只验证正确性，性能结论见 Part 3。
