import torch


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Reference RMSNorm with FP32 accumulation."""

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")

    if weight.ndim != 1:
        raise ValueError("weight must be one-dimensional")

    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"hidden size mismatch: x={x.shape[-1]}, weight={weight.shape[0]}"
        )

    output_dtype = x.dtype #保存数据类型

    # TODO 1：把 x 转成 FP32
    x_fp32 = x.float()
    # TODO 2：计算最后一维平方均值，保留被规约的维度
    mean_square = x_fp32.square().mean(dim=-1, keepdim=True)
    # TODO 3：计算 inverse RMS，并完成 x、inverse RMS、weight 的乘法
    inverse_rms = torch.rsqrt(mean_square + eps)
    # TODO 4：把结果转换回输入 dtype
    output = (x_fp32 * inverse_rms * weight.float()).to(output_dtype)
    return output

if __name__ == "__main__":
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    weight = torch.tensor([2.0, 1.0], dtype=torch.float32)
    eps = 0.0

    x_fp32 = x.float()
    squared = x_fp32.square()
    mean_square = squared.mean(dim=-1, keepdim=True)
    # 平方根倒数rsqrt
    inverse_rms = torch.rsqrt(mean_square + eps)
    output = rmsnorm_reference(x, weight, eps)

    print("x:", x)
    print("x shape:", x.shape)
    print("squared:", squared)
    print("mean_square:", mean_square)
    print("mean_square shape:", mean_square.shape)
    print("inverse_rms:", inverse_rms)
    print("output:", output)
    print("output shape:", output.shape)

    print("\nFP16 test")

    x_fp16 = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
    weight_fp16 = torch.tensor([2.0, 1.0], dtype=torch.float16)

    output_fp16 = rmsnorm_reference(
        x_fp16,
        weight_fp16,
        eps=0.0,
    )

    print("input dtype:", x_fp16.dtype)
    print("weight dtype:", weight_fp16.dtype)
    print("accumulation dtype:", x_fp16.float().dtype)
    print("output dtype:", output_fp16.dtype)
    print("FP16 output:", output_fp16)
    print("FP32 output:", output)
    print(
        "absolute difference:",
        (output_fp16.float() - output).abs(),
    )
    # 自动检查
    torch.testing.assert_close(
        output_fp16.float(),
        output,
        atol=1.0e-3,
        rtol=1.0e-3,
    )

    print("fp16_close_to_fp32=PASS")