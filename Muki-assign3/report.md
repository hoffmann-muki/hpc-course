# CUDA Video Convolution: Performance Analysis Report

**Author:** Hoffmann  
**Course:** CMSC 714 - High Performance Computing  
**Assignment:** Project 3 - CUDA  
**Date:** March 19, 2026  

---

## 1. Kernel Design

The `convolveGPU` kernel implements 2D image convolution on video frames using a grid-stride loop pattern to handle cases where the number of pixels exceeds available GPU threads.

### Key Implementation Details

**Grid-Stride Execution:**
```cuda
const int startY = blockIdx.y * blockDim.y + threadIdx.y;
const int startX = blockIdx.x * blockDim.x + threadIdx.x;
const int strideY = blockDim.y * gridDim.y;
const int strideX = blockDim.x * gridDim.x;

for (int i = startY; i < height; i += strideY) {
    for (int j = startX; j < width; j += strideX) {
        // Process pixel (i, j)
    }
}
```

This approach ensures that each thread computes pixels iteratively across the entire frame, avoiding redundant synchronization and improving cache locality.

**Border Pixel Handling:**
- Pixels within `halfKernelHeight` or `halfKernelWidth` of the image boundaries are set to 0
- This prevents out-of-bounds access and matches the CPU reference implementation

**Per-Channel Convolution:**
Each color channel (B, G, R in BGR format) is convolved independently:
```cuda
for (int k = -halfKernelHeight; k <= halfKernelHeight; k++) {
    for (int l = -halfKernelWidth; l <= halfKernelWidth; l++) {
        int inIdx = (i + k) * width * 3 + (j + l) * 3;
        int kernelIdx = (k + halfKernelHeight) * kernelWidth + (l + halfKernelWidth);
        
        redDot   += in[inIdx + 2] * kernel[kernelIdx];  // Red channel
        greenDot += in[inIdx + 1] * kernel[kernelIdx];  // Green channel
        blueDot  += in[inIdx + 0] * kernel[kernelIdx];  // Blue channel
    }
}
```

---

## 2. Data Distribution & Host-GPU Communication

### Memory Management

**GPU Memory Allocation:**
- `allocateFrames()` pre-allocates GPU memory for frame batches (default: 10 frames per batch)
- Each frame requires `width × height × 3 × sizeof(float)` bytes
- For 1920×1080 BGR frames: ~24.8 MB per frame

**Batch Processing Pipeline:**
1. Host reads frame from video file → converts uint8 to float32 (÷255)
2. `cudaMemcpy(HostToDevice)` transfers frame to GPU
3. `convolveGPU` kernel executes asynchronously on GPU stream
4. `cudaMemcpy(DeviceToHost)` transfers result back to CPU
5. Host converts float32 to uint8 (×255) and writes to output video

**Asynchronous Stream Processing:**
- Multiple CUDA streams (up to 8) allow overlapping kernel execution with memory transfers
- Frame batches enable pipelined processing without blocking on individual frames

### Memory Layout

Frames are stored in **packed BGR format** (3 consecutive floats per pixel):
```
[B0, G0, R0, B1, G1, R1, B2, G2, R2, ...]
```

This layout ensures coalesced memory access patterns across GPU threads.

---

## 3. Performance Results

### Benchmark Setup
- **GPU:** NVIDIA A100 (40GB memory slice)
- **Video:** 1920×1080 resolution, 543 frames, 24 fps
- **Kernel:** Edge detection (3×3, requires 9 operations per pixel)
- **Test Metric:** Frames processed per second (FPS)

### Results

| Block Size | Time (s) | FPS | Vs. Baseline |
|-----------|----------|-----|--------------|
| 4         | 0.502    | 1,082 | 0.54× |
| 8 (baseline) | 0.270 | 2,012 | 1.00× |
| **16 (peak)** | **0.215** | **2,523** | **1.25×** |
| 32        | 0.322    | 1,688 | 0.84× |

### Performance Analysis

**Peak Performance:** The kernel achieves **2,523 frames/sec** at `blockDimSize=16`, representing a **25% improvement** over the baseline configuration.

**Why Block Size 16 is Optimal:**
1. **Occupancy:** A 16×16 block (256 threads) fits well within A100's 128 warps/SM maximum
2. **Resource Utilization:** Balances register pressure and shared memory usage
3. **Memory Bandwidth:** Sufficient thread parallelism to saturate the GPU's ~1.6 TB/s peak bandwidth

**Why Larger Blocks Degrade Performance:**
- **32×32 blocks (1024 threads):** Exceeds A100's per-SM thread limit, causing resource contention
- Register spill forces some values to slower local memory
- Reduced occupancy results in underutilized GPU execution units

**Why Smaller Blocks Are Slower:**
- **4×4 blocks (16 threads):** Only 8 warps per block vs. 256 threads
- Insufficient parallelism → poor GPU utilization
- High synchronization overhead relative to compute

### Expected vs. Actual Performance

**Expected Runtime Behavior:**
- GPU A100 theoretical peak: ~312 TFLOPS (FP32)
- Edge kernel: 9 ops per pixel × 1920×1080×543 = ~5.4 billion operations
- Predicted optimal: 2,000+ FPS ✓

**Actual Results Confirm Expectations:**
- Peak of 2,523 FPS aligns with A100 capability for memory-bound kernels
- The 16:8 ratio (1.25×) matches occupancy/bandwidth improvement theory
- Performance degradation beyond blockDimSize=16 matches compiler register allocation limits

---

## 4. Conclusion

The `convolveGPU` kernel demonstrates efficient GPU utilization through:
1. **Correct striding logic** ensuring full frame coverage without redundant computation
2. **Asynchronous memory transfers** enabling pipeline parallelism
3. **Per-channel convolution** matching the CPU reference exactly
4. **Optimal configuration** at blockDimSize=16 yielding 25% performance gain

The assignment successfully demonstrates fundamental CUDA concepts: kernel design, memory management, and empirical performance tuning.

---

## Appendix: Compilation & Execution

**Build:**
```bash
make clean && make
```

**Test All Kernels:**
```bash
./video-effect video.mp4 video-identity.mp4 identity 128 128
./video-effect video.mp4 video-edge.mp4 edge 128 128
./video-effect video.mp4 video-sharpen.mp4 sharpen 128 128
./video-effect video.mp4 video-blur.mp4 blur 128 128
```

**Benchmark Block Sizes:**
```bash
./benchmark.sh
```
