"""Inference-only FP32 CUDA U(Vx), compiled with NVRTC bundled by PyTorch.

No activation, dense reconstruction, CPU inference, or new fitted weights.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SOURCE = r'''
extern "C" __global__ void lowrank_fused(
    const float* __restrict__ x, const float* __restrict__ v,
    const float* __restrict__ u, float* __restrict__ y,
    int k, int r, int n, int tile) {
    extern __shared__ float z[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;
    const int row = blockIdx.x;
    const int first = blockIdx.y * tile;
    // Warp-coalesced first projection; latent vector lives only in shared memory.
    for (int j = warp; j < r; j += warps) {
        float sum = 0.0f;
        for (int c = lane; c < k; c += 32)
            sum = fmaf(x[row * k + c], v[j * k + c], sum);
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) z[j] = sum;
    }
    __syncthreads();
    // Blocks owning different output tiles independently recompute the latent vector.
    for (int j = first + warp; j < first + tile && j < n; j += warps) {
        float sum = 0.0f;
        for (int c = lane; c < r; c += 32)
            sum = fmaf(z[c], u[j * r + c], sum);
        for (int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if (lane == 0) y[row * n + j] = sum;
    }
}
'''


class Runtime:
    def __init__(self, device):
        if os.name != "nt":
            raise RuntimeError("This tested NVRTC loader currently supports Windows only")
        self.device = torch.device(device)
        torch.cuda.init()
        lib = Path(torch.__file__).parent / "lib"
        dlls = list(lib.glob("nvrtc64_*.dll"))
        if len(dlls) != 1:
            raise RuntimeError("Expected exactly one bundled NVRTC DLL")
        self.nvrtc = C.CDLL(str(dlls[0]))
        self.driver = C.WinDLL("nvcuda.dll")
        self.nvrtc.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p,
                                                C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)]
        self.nvrtc.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
        for name in ("nvrtcGetProgramLogSize", "nvrtcGetPTXSize"):
            getattr(self.nvrtc, name).argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
        for name in ("nvrtcGetProgramLog", "nvrtcGetPTX"):
            getattr(self.nvrtc, name).argtypes = [C.c_void_p, C.c_void_p]
        self.nvrtc.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
        self.driver.cuModuleLoadDataEx.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint,
                                                  C.c_void_p, C.c_void_p]
        self.driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        self.driver.cuLaunchKernel.argtypes = [C.c_void_p, *([C.c_uint] * 7), C.c_void_p,
                                               C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        self.driver.cuGetErrorName.argtypes = [C.c_int, C.POINTER(C.c_char_p)]
        with torch.cuda.device(self.device):
            major, minor = torch.cuda.get_device_capability(self.device)
            self.options = [f"--gpu-architecture=compute_{major}{minor}".encode(), b"--std=c++11"]
            program = C.c_void_p()
            self._nvcheck(self.nvrtc.nvrtcCreateProgram(C.byref(program), SOURCE.encode(),
                                                      b"gpu_lowrank.cu", 0, None, None))
            try:
                options = (C.c_char_p * len(self.options))(*self.options)
                result = self.nvrtc.nvrtcCompileProgram(program, len(options), options)
                size = C.c_size_t()
                self._nvcheck(self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size)))
                log = C.create_string_buffer(size.value)
                self._nvcheck(self.nvrtc.nvrtcGetProgramLog(program, log))
                self.compile_log = log.value.decode(errors="replace")
                if result:
                    raise RuntimeError(f"NVRTC error {result}: {self.compile_log}")
                self._nvcheck(self.nvrtc.nvrtcGetPTXSize(program, C.byref(size)))
                ptx = C.create_string_buffer(size.value)
                self._nvcheck(self.nvrtc.nvrtcGetPTX(program, ptx))
                self.ptx = ptx.raw
            finally:
                self.nvrtc.nvrtcDestroyProgram(C.byref(program))
            self.module, self.function = C.c_void_p(), C.c_void_p()
            self._check(self.driver.cuModuleLoadDataEx(C.byref(self.module), ptx, 0, None, None))
            self._check(self.driver.cuModuleGetFunction(C.byref(self.function), self.module, b"lowrank_fused"))

    @staticmethod
    def _nvcheck(code):
        if code:
            raise RuntimeError(f"NVRTC API error {code}")

    def _check(self, code):
        if code:
            name = C.c_char_p()
            self.driver.cuGetErrorName(code, C.byref(name))
            raise RuntimeError(f"CUDA driver error {code}: {name.value}")

    def launch(self, x, v, u, y, tile):
        k, r, n = x.shape[-1], v.shape[0], u.shape[0]
        rows = x.numel() // k
        args = [C.c_void_p(t.data_ptr()) for t in (x, v, u, y)]
        args += [C.c_int(value) for value in (k, r, n, tile)]
        pointers = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device).cuda_stream
            self._check(self.driver.cuLaunchKernel(self.function, rows, (n + tile - 1) // tile, 1,
                        256, 1, 1, r * 4, C.c_void_p(stream), pointers, None))


_runtimes = {}


def runtime(device):
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _runtimes:
        _runtimes[index] = Runtime(torch.device("cuda", index))
    return _runtimes[index]


def fused_lowrank(x, v, u, *, tile=128):
    """Compute x @ v.T @ u.T on the current CUDA stream; FP32 inference only."""
    if not all(t.is_cuda for t in (x, v, u)):
        raise ValueError("All tensors must be CUDA tensors; no CPU fallback")
    if any(t.dtype != torch.float32 for t in (x, v, u)):
        raise ValueError("Only FP32 is supported")
    if x.ndim < 1 or v.ndim != 2 or u.ndim != 2 or x.shape[-1] != v.shape[1] or v.shape[0] != u.shape[1]:
        raise ValueError("Expected x[...,K], v[R,K], u[N,R]")
    if len({t.device for t in (x, v, u)}) != 1:
        raise ValueError("All tensors must share a device")
    if not all(t.is_contiguous() for t in (x, v, u)):
        raise ValueError("Contiguous tensors required")
    if any(t.requires_grad for t in (x, v, u)) and torch.is_grad_enabled():
        raise ValueError("Inference only; autograd is not implemented")
    if not 0 < v.shape[0] <= 12288 or not 0 < tile <= 4096 or min(v.shape) <= 0 or u.shape[0] <= 0:
        raise ValueError("Invalid size or shared-memory budget")
    if x.numel() > 2**31 - 1 or u.numel() > 2**31 - 1 or v.numel() > 2**31 - 1:
        raise ValueError("32-bit kernel indexing limit exceeded")
    y = torch.empty((*x.shape[:-1], u.shape[0]), dtype=x.dtype, device=x.device)
    if x.numel():
        runtime(x.device).launch(x, v, u, y, tile)
    return y


class GpuLowRank(nn.Module):
    """Same factors, custom decode kernel; optionally use GPU cuBLAS for prefill."""
    def __init__(self, v, u, *, tile=128, decode_only=False):
        super().__init__()
        if not (v.is_cuda and u.is_cuda):
            raise ValueError("Construct this layer with CUDA factors")
        self.register_buffer("v", v.detach().contiguous())
        self.register_buffer("u", u.detach().contiguous())
        self.tile, self.decode_only = tile, decode_only

    def forward(self, x):
        if not x.is_cuda:
            raise ValueError("GPU-only inference")
        if self.decode_only and x.numel() // x.shape[-1] > 1:
            return F.linear(F.linear(x, self.v), self.u)
        return fused_lowrank(x.contiguous(), self.v, self.u, tile=self.tile)


def source_hash():
    return hashlib.sha256(SOURCE.encode()).hexdigest()
