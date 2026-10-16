"""GPU-only U(Vx) with one intermediate computation and optional graph replay.

The earlier kernel remains immutable. These are inference execution alternatives,
not changes to the approximation. Prefill uses existing GPU cuBLAS operations.
"""
import ctypes as C
import threading

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .gpu_lowrank import Runtime
else:
    from gpu_lowrank import Runtime

SOURCE = r'''
extern "C" __global__ void project_once(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y, int k, int n) {
    const int lane = threadIdx.x & 31;
    const int output = blockIdx.y * 8 + (threadIdx.x >> 5);
    if (output >= n) return;
    float sum = 0.0f;
    for (int c = lane; c < k; c += 32)
        sum = fmaf(x[blockIdx.x * k + c], w[output * k + c], sum);
    for (int offset = 16; offset; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0) y[blockIdx.x * n + output] = sum;
}
'''


class StagedRuntime(Runtime):
    def __init__(self, device):
        super().__init__(device)
        with torch.cuda.device(self.device):
            program = C.c_void_p()
            self._nvcheck(self.nvrtc.nvrtcCreateProgram(C.byref(program), SOURCE.encode(),
                                                      b"gpu_lowrank_v2.cu", 0, None, None))
            try:
                options = (C.c_char_p * len(self.options))(*self.options)
                code = self.nvrtc.nvrtcCompileProgram(program, len(options), options)
                size = C.c_size_t()
                self._nvcheck(self.nvrtc.nvrtcGetProgramLogSize(program, C.byref(size)))
                log = C.create_string_buffer(size.value)
                self._nvcheck(self.nvrtc.nvrtcGetProgramLog(program, log))
                if code:
                    raise RuntimeError(log.value.decode(errors="replace"))
                self._nvcheck(self.nvrtc.nvrtcGetPTXSize(program, C.byref(size)))
                ptx = C.create_string_buffer(size.value)
                self._nvcheck(self.nvrtc.nvrtcGetPTX(program, ptx))
                self.staged_ptx = ptx.raw
            finally:
                self.nvrtc.nvrtcDestroyProgram(C.byref(program))
            self.staged_module, self.project = C.c_void_p(), C.c_void_p()
            self._check(self.driver.cuModuleLoadDataEx(C.byref(self.staged_module), ptx, 0, None, None))
            self._check(self.driver.cuModuleGetFunction(C.byref(self.project), self.staged_module, b"project_once"))

    def project_into(self, x, w, y):
        k, n = w.shape[1], w.shape[0]
        args = [C.c_void_p(t.data_ptr()) for t in (x, w, y)] + [C.c_int(k), C.c_int(n)]
        pointers = (C.c_void_p * len(args))(*[C.cast(C.pointer(a), C.c_void_p) for a in args])
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self._check(self.driver.cuLaunchKernel(self.project, x.numel() // k, (n + 7) // 8, 1,
                    256, 1, 1, 0, C.c_void_p(stream), pointers, None))


_staged_runtimes = {}


def staged_runtime(device):
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _staged_runtimes:
        _staged_runtimes[index] = StagedRuntime(torch.device("cuda", index))
    return _staged_runtimes[index]


def validate(x, v, u):
    if not all(t.is_cuda and t.dtype == torch.float32 for t in (x, v, u)):
        raise ValueError("FP32 CUDA tensors required; no CPU fallback")
    if x.ndim < 1 or v.ndim != 2 or u.ndim != 2 or x.shape[-1] != v.shape[1] or v.shape[0] != u.shape[1]:
        raise ValueError("Expected x[...,K], v[R,K], u[N,R]")
    if len({t.device for t in (x, v, u)}) != 1 or min(*v.shape, *u.shape) <= 0:
        raise ValueError("Invalid shape or device")
    if any(t.requires_grad for t in (x, v, u)) and torch.is_grad_enabled():
        raise ValueError("Inference only; autograd not implemented")
    if max(t.numel() for t in (x, v, u)) > 2**31 - 1 or u.shape[0] > 8 * 65535 or v.shape[0] > 8 * 65535:
        raise ValueError("CUDA indexing/grid limit exceeded")


def staged_lowrank(x, v, u):
    validate(x, v, u)
    x, v, u = x.contiguous(), v.contiguous(), u.contiguous()
    z = torch.empty((*x.shape[:-1], v.shape[0]), device=x.device, dtype=x.dtype)
    y = torch.empty((*x.shape[:-1], u.shape[0]), device=x.device, dtype=x.dtype)
    if x.numel():
        with torch.cuda.device(x.device):
            rt = staged_runtime(x.device)
            rt.project_into(x, v, z)
            rt.project_into(z, u, y)
    return y


class ReusedGpuLowRank(nn.Module):
    """Decode mode: staged, graph_staged, graph_cublas. CUDA cuBLAS for prefill.

    Graphs retain one-row buffers per stream and clone returned outputs so later
    replays cannot overwrite caller-held results. No mutable output alias escapes.
    Factor buffers must not be replaced or edited after graph preparation.
    """
    def __init__(self, v, u, mode="staged"):
        super().__init__()
        if mode not in ("staged", "graph_staged", "graph_cublas"):
            raise ValueError("Unsupported execution mode")
        validate(torch.empty((0, v.shape[1]), device=v.device, dtype=v.dtype), v, u)
        self.register_buffer("v", v.detach().contiguous())
        self.register_buffer("u", u.detach().contiguous())
        self.mode, self.graphs = mode, {}
        self._lock = threading.RLock()

    def _prepare(self, stream_key):
        static_x = torch.zeros((1, self.v.shape[1]), device=self.v.device)
        static_z = torch.empty((1, self.v.shape[0]), device=self.v.device)
        static_y = torch.empty((1, self.u.shape[0]), device=self.v.device)
        rt = staged_runtime(self.v.device) if self.mode == "graph_staged" else None
        def operation():
            if rt is None:
                torch.mm(static_x, self.v.T, out=static_z)
                torch.mm(static_z, self.u.T, out=static_y)
            else:
                rt.project_into(static_x, self.v, static_z)
                rt.project_into(static_z, self.u, static_y)
        current = torch.cuda.current_stream(self.v.device)
        side = torch.cuda.Stream(device=self.v.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            for _ in range(3):
                operation()
        current.wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            operation()
        self.graphs[stream_key] = (static_x, static_z, static_y, graph)

    def forward(self, x):
        validate(x, self.v, self.u)
        if x.numel() // x.shape[-1] != 1:
            return F.linear(F.linear(x, self.v), self.u)
        if self.mode == "staged":
            return staged_lowrank(x, self.v, self.u)
        with torch.cuda.device(x.device), self._lock:
            key = torch.cuda.current_stream(x.device).cuda_stream
            if key not in self.graphs:
                self._prepare(key)
            static_x, _, static_y, graph = self.graphs[key]
            static_x.copy_(x.reshape_as(static_x))
            graph.replay()
            return static_y.clone().reshape(*x.shape[:-1], self.u.shape[0])
