import pytest
import torch
from src.gpu_lowrank_v2 import ReusedGpuLowRank, staged_lowrank

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@pytest.mark.parametrize("rows,k,r,n", [(1, 1024, 275, 640), (7, 37, 11, 23), (128, 65, 33, 67)])
def test_staged_matches_reference(rows, k, r, n):
    torch.manual_seed(11)
    x, v, u = [torch.randn(shape, device="cuda") for shape in ((rows, k), (r, k), (n, r))]
    expected = (x @ v.T) @ u.T
    actual = staged_lowrank(x, v, u)
    assert (actual - expected).norm() / expected.norm() < 2e-6


@pytest.mark.parametrize("mode", ["staged", "graph_staged", "graph_cublas"])
def test_decode_parity_no_output_alias_and_prefill(mode):
    x = torch.randn(1, 1, 1024, device="cuda")
    v, u = torch.randn(275, 1024, device="cuda"), torch.randn(640, 275, device="cuda")
    layer = ReusedGpuLowRank(v, u, mode).eval()
    a = layer(x)
    retained = a.clone()
    b = layer(x * 2)
    expected = (x @ v.T) @ u.T
    assert (a - expected).norm() / expected.norm() < 2e-6
    assert (b - 2 * expected).norm() / (2 * expected).norm() < 2e-6
    assert torch.equal(a, retained)
    prefill = torch.randn(1, 128, 1024, device="cuda")
    assert torch.equal(layer(prefill), (prefill @ v.T) @ u.T)


def test_graph_isolated_across_streams():
    v, u = torch.randn(11, 37, device="cuda"), torch.randn(23, 11, device="cuda")
    layer = ReusedGpuLowRank(v, u, "graph_staged")
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for stream in streams:
        with torch.cuda.stream(stream):
            x = torch.randn(1, 37, device="cuda")
            results.append((layer(x), (x @ v.T) @ u.T))
    for stream in streams:
        stream.synchronize()
    assert len(layer.graphs) == 2
    assert all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in results)
