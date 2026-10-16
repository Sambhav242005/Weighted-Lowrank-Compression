
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
