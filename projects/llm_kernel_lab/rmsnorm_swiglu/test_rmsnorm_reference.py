import torch
import torch.nn.functional as functional

from rmsnorm_reference import rmsnorm_reference


def test_float32_against_pytorch() -> None:
    torch.manual_seed(42)

    rows = 3
    hidden_size = 7
    eps = 1.0e-5

    x = torch.randn(rows, hidden_size, dtype=torch.float32)
    weight = torch.randn(hidden_size, dtype=torch.float32)

    actual = rmsnorm_reference(x, weight, eps)

    expected = functional.rms_norm(
        x,
        normalized_shape=(hidden_size,),
        weight=weight,
        eps=eps,
    )

    torch.testing.assert_close(
        actual,
        expected,
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    print("float32_against_pytorch=PASS")
    print("shape:", actual.shape)
    print("dtype:", actual.dtype)
    print("max_absolute_error:", (actual - expected).abs().max().item())


if __name__ == "__main__":
    test_float32_against_pytorch()