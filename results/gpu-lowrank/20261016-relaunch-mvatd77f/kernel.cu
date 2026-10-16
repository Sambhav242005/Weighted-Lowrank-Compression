
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
