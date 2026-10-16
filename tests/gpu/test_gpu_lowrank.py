"""Opt-in CUDA numerical, tail, stream and contract checks; no model downloads."""
import pytest
import torch
import subprocess
import sys
from src.gpu_lowrank import fused_lowrank

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_compile_before_first_cuda_tensor_in_fresh_process():
    subprocess.run(["rtk", sys.executable, "-c",
        "import torch; from src.gpu_lowrank import runtime; "
        "runtime(torch.device('cuda',0)); print('fresh CUDA startup passed')"], check=True)


@pytest.mark.parametrize("rows,k,r,n,tile", [(1, 1024, 275, 640, 128), (1, 37, 11, 23, 64),
    (7, 65, 33, 67, 32), (128, 1024, 275, 640, 128), (1, 1024, 275, 640, 640)])
def test_matches_two_linear_projections(rows, k, r, n, tile):
    torch.manual_seed(11)
    x, v, u = [torch.randn(shape, device="cuda") for shape in ((rows, k), (r, k), (n, r))]
    expected = (x @ v.T) @ u.T
    actual = fused_lowrank(x, v, u, tile=tile)
    torch.cuda.synchronize()
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all()
    assert torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected) < 2e-6


def test_current_stream_empty_input_and_dtype_guard():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, v, u = [torch.randn(shape, device="cuda") for shape in ((3, 37), (11, 37), (23, 11))]
        actual = fused_lowrank(x, v, u)
        expected = (x @ v.T) @ u.T
    stream.synchronize()
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert fused_lowrank(x[:0], v, u).shape == (0, 23)
    with pytest.raises(ValueError, match="FP32"):
        fused_lowrank(x.half(), v.half(), u.half())
    with pytest.raises(ValueError, match="Expected"):
        fused_lowrank(x, v[:, :2].contiguous(), u)
