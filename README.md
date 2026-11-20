# Experiment 6: Memory Optimization in CUDA

Coalesced vs. uncoalesced global-memory access, naive matrix multiplication, and execution-time comparison on an NVIDIA GPU.

---

## Table of Contents

1. [Aim](#1-aim)
2. [Theory](#2-theory)
3. [Environment and Setup](#3-environment-and-setup)
4. [Project Structure](#4-project-structure)
5. [Implementation](#5-implementation)
6. [Compile and Run](#6-compile-and-run)
7. [Results](#7-results)
8. [Analysis and Discussion](#8-analysis-and-discussion)
9. [Conclusion](#9-conclusion)

---

## 1. Aim

To study how the global-memory access pattern of a CUDA kernel affects its performance by:

- comparing an **uncoalesced** and a **coalesced** (shared-memory tiled) matrix transpose,
- running a **naive matrix multiplication** kernel, and
- comparing the execution times measured on the GPU using CUDA events.

---

## 2. Theory

### 2.1 Memory coalescing

GPU threads execute in groups of 32 called **warps**. When the 32 threads of a warp access neighbouring addresses in global memory, the hardware merges them into a few wide memory transactions. When the addresses are far apart (large stride), the warp needs many transactions and most of the fetched bytes are wasted, so effective bandwidth drops.

| | Coalesced | Uncoalesced |
|---|---|---|
| Thread *i* of a warp accesses | `base + i` | `base + i × N` (stride N) |
| Memory transactions per warp | Few (1 to 4) | Up to 32 |
| Bandwidth used | Close to peak | A small fraction of peak |

### 2.2 Matrix transpose

Transpose computes `out[j][i] = in[i][j]`. In row-major storage, element `(r, c)` is at index `r*N + c`.

- **Uncoalesced kernel:** each thread copies one element. Consecutive threads read consecutive elements of a row (good) but write to elements `N` apart in a column (bad).
- **Coalesced kernel:** a block first loads a 32 × 32 tile into shared memory with coalesced reads, synchronises, then writes the tile out transposed, again with coalesced writes. The shared array is declared `[32][33]`; the extra column (padding) avoids shared-memory bank conflicts.

### 2.3 Naive matrix multiplication

For `C = A × B` with `N × N` matrices, `C[i][j] = Σ_k A[i][k] * B[k][j]`. In the naive kernel one thread computes one element of `C` with a loop of `N` iterations; every iteration does 2 global-memory reads and 1 multiply-add. Nothing is reused through shared memory, so the same data is fetched from global memory again and again. This is the baseline that shared-memory tiling improves on.

### 2.4 Formulas used

| Quantity | Formula |
|---|---|
| Effective bandwidth (transpose) | `2 × N × N × 4 bytes / time` (one read + one write of every float) |
| Floating-point rate (matmul) | `2 × N³ operations / time` (one multiply + one add per k) |
| Speedup | `time (uncoalesced) / time (coalesced)` |

---

## 3. Environment and Setup

### 3.1 System used

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB, 16311 MiB reported) |
| Compute capability | 12.0 (`sm_120`) |
| NVIDIA driver | 610.88 (WDDM) |
| CUDA version reported by driver | 13.3 |
| CUDA compiler (`nvcc`) | release 13.4, V13.4.92 |
| Host compiler | Microsoft Visual Studio Build Tools (x64) |
| OS | Windows 11 |

### 3.2 Verify the tools

Open the **x64 Native Tools Command Prompt for VS** (so that `cl.exe` is on the PATH) and run:

```bat
nvidia-smi
nvcc --version
```

`nvidia-smi` shows the GPU name, driver version and CUDA version; `nvcc --version` shows the CUDA compiler version.

> 📸 **Screenshot: environment check **
<img width="1600" height="900" alt="WhatsApp Image 2026-11-20 at 10 04 27 AM" src="https://github.com/user-attachments/assets/d0d83a98-1936-46eb-9e33-0abed753ef6f" />


### 3.3 Create the working folder

```bat
cd %USERPROFILE%\Documents
mkdir Experiment6_CUDA_Memory_Optimization
cd Experiment6_CUDA_Memory_Optimization
notepad exp6.cu
```

### 3.4 Alternative: Google Colab

If no local GPU is available: open [colab.research.google.com](https://colab.research.google.com), choose **Runtime → Change runtime type → GPU**, run `!nvidia-smi` to confirm, create the file with `%%writefile exp6.cu`, and compile with `!nvcc -O2 -o exp6 exp6.cu` and run with `!./exp6`.

---

## 4. Project Structure

```
Experiment6_CUDA_Memory_Optimization/
├── exp6.cu                  # complete CUDA source (transpose x2 + naive matmul)
├── exp6.exe                 # compiled binary (Windows)
├── README.md                # this file
└── screenshots/
    ├── 01_environment_check.png
    ├── 02_compile_error_and_fix.png
    └── 03_program_output_runs.png
```

---

## 5. Implementation

`exp6.cu` is written in five parts and pasted one after another into the same file.

### Step 1: Headers and error-check macro

The `CHECK` macro stops the program with a message if any CUDA call fails. `TILE` and `BLK` are the block dimensions.

```cuda
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32   // transpose block = 32 x 32 threads
#define BLK  16   // matmul block   = 16 x 16 threads

#define CHECK(call) do { cudaError_t e = (call);                    \
    if (e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",                 \
                cudaGetErrorString(e), __FILE__, __LINE__);         \
        exit(1); } } while (0)
```

### Step 2: The two transpose kernels

The second kernel stages data in `tile[][]`, calls `__syncthreads()`, and swaps the block indices before writing.

```cuda
// ---------- Kernel 1: UNCOALESCED transpose (one element per thread) ----------
__global__ void transposeNaive(const float *in, float *out, int N)
{
    int x = blockIdx.x * TILE + threadIdx.x;   // column index
    int y = blockIdx.y * TILE + threadIdx.y;   // row index
    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];        // read: row-wise (good)
                                               // write: column-wise, stride N (bad)
}

// ---------- Kernel 2: COALESCED transpose (shared-memory tile) ----------
__global__ void transposeCoalesced(const float *in, float *out, int N)
{
    __shared__ float tile[TILE][TILE + 1];     // +1 padding avoids bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // coalesced read

    __syncthreads();                           // wait until the tile is loaded

    x = blockIdx.y * TILE + threadIdx.x;       // swap block coordinates
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // coalesced write
}
```

### Step 3: The naive matrix-multiplication kernel

Thread `(row, col)` owns one element of `C`. The grid is 2D with 16 × 16 blocks; `dim3 block(16,16)` and `dim3 grid(N/16, N/16)` are set in the host code.

```cuda
// ---------- Kernel 3: NAIVE matrix multiplication C = A x B ----------
__global__ void matMulNaive(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)            // 2 global reads + 1 multiply-add
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}
```

### Step 4: Host helpers (timing and verification)

Kernel launches are asynchronous, so the program times them with **CUDA events** and not a CPU clock alone. The first launch is a warm-up and is not timed. The transpose is repeated 20 times and the average is reported. The checkers compare the GPU output with a CPU reference.

```cuda
// ---------- Host helpers: timing with CUDA events, and result checking ----------
typedef void (*TKernel)(const float *, float *, int);

float timeTranspose(TKernel k, dim3 g, dim3 b, const float *in, float *out,
                    int N, int reps)
{
    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    CHECK(cudaMemset(out, 0, (size_t)N * N * sizeof(float)));
    k<<<g, b>>>(in, out, N);                       // warm-up launch (not timed)
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(s));
    for (int r = 0; r < reps; r++) k<<<g, b>>>(in, out, N);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
    CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    return ms / reps;                              // average time per launch
}

bool checkTranspose(const float *in, const float *out, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (out[j * N + i] != in[i * N + j]) return false;
    return true;
}

bool checkMatMul(const float *A, const float *B, const float *C, int N)
{
    for (int t = 0; t < 20; t++) {                 // spot-check 20 random elements
        int i = rand() % N, j = rand() % N;
        double ref = 0.0;
        for (int k = 0; k < N; k++) ref += (double)A[i * N + k] * B[k * N + j];
        if (fabs(ref - C[i * N + j]) > 1e-3 * fabs(ref) + 1e-3) return false;
    }
    return true;
}
```

### Step 5: `main()`

`main` allocates host and device memory, copies the input to the GPU, launches and times each kernel, copies the result back, verifies it, and prints the table. Default sizes are **N = 4096** for the transpose and **N = 1024** for the matrix multiplication; both can be changed from the command line.

```cuda
int main(int argc, char **argv)
{
    int NT = 4096;                       // transpose size (NT x NT)
    int NM = 1024;                       // matmul size    (NM x NM)
    if (argc > 1) NT = atoi(argv[1]);
    if (argc > 2) NM = atoi(argv[2]);

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);

    // ================= PART A : TRANSPOSE =================
    size_t bytesT = (size_t)NT * NT * sizeof(float);
    float *h_in  = (float *)malloc(bytesT);
    float *h_out = (float *)malloc(bytesT);
    for (size_t i = 0; i < (size_t)NT * NT; i++) h_in[i] = (float)(i % 1000);

    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in,  bytesT));
    CHECK(cudaMalloc(&d_out, bytesT));
    CHECK(cudaMemcpy(d_in, h_in, bytesT, cudaMemcpyHostToDevice));

    dim3 blockT(TILE, TILE);
    dim3 gridT((NT + TILE - 1) / TILE, (NT + TILE - 1) / TILE);

    float tN = timeTranspose(transposeNaive, gridT, blockT, d_in, d_out, NT, 20);
    CHECK(cudaMemcpy(h_out, d_out, bytesT, cudaMemcpyDeviceToHost));
    bool okN = checkTranspose(h_in, h_out, NT);

    float tC = timeTranspose(transposeCoalesced, gridT, blockT, d_in, d_out, NT, 20);
    CHECK(cudaMemcpy(h_out, d_out, bytesT, cudaMemcpyDeviceToHost));
    bool okC = checkTranspose(h_in, h_out, NT);

    double gbN = 2.0 * bytesT / (tN * 1e6);          // read + write, GB/s
    double gbC = 2.0 * bytesT / (tC * 1e6);

    printf("\n=== PART A: Matrix transpose (N = %d) ===\n", NT);
    printf("Uncoalesced : %8.3f ms | %7.1f GB/s | %s\n", tN, gbN, okN ? "PASS" : "FAIL");
    printf("Coalesced   : %8.3f ms | %7.1f GB/s | %s\n", tC, gbC, okC ? "PASS" : "FAIL");
    printf("Speedup (uncoalesced / coalesced) = %.2fx\n", tN / tC);

    // ================= PART B : NAIVE MATMUL =================
    size_t bytesM = (size_t)NM * NM * sizeof(float);
    float *hA = (float *)malloc(bytesM);
    float *hB = (float *)malloc(bytesM);
    float *hC = (float *)malloc(bytesM);
    for (size_t i = 0; i < (size_t)NM * NM; i++) {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytesM));
    CHECK(cudaMalloc(&dB, bytesM));
    CHECK(cudaMalloc(&dC, bytesM));
    CHECK(cudaMemcpy(dA, hA, bytesM, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytesM, cudaMemcpyHostToDevice));

    dim3 block(BLK, BLK);
    dim3 grid((NM + BLK - 1) / BLK, (NM + BLK - 1) / BLK);

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
    matMulNaive<<<grid, block>>>(dA, dB, dC, NM);    // warm-up
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    int reps = 5;
    CHECK(cudaEventRecord(s));
    for (int r = 0; r < reps; r++) matMulNaive<<<grid, block>>>(dA, dB, dC, NM);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
    float tM = ms / reps;

    CHECK(cudaMemcpy(hC, dC, bytesM, cudaMemcpyDeviceToHost));
    bool okM = checkMatMul(hA, hB, hC, NM);
    double gflops = 2.0 * NM * NM * (double)NM / (tM * 1e6);

    printf("\n=== PART B: Naive matrix multiplication (N = %d) ===\n", NM);
    printf("Naive matmul: %8.3f ms | %7.1f GFLOPS | %s\n", tM, gflops, okM ? "PASS" : "FAIL");

    // ================= SUMMARY =================
    printf("\n=== SUMMARY ===\n");
    printf("Transpose uncoalesced : %.3f ms\n", tN);
    printf("Transpose coalesced   : %.3f ms\n", tC);
    printf("Naive matmul          : %.3f ms\n", tM);

    cudaFree(d_in); cudaFree(d_out); cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(h_in); free(h_out); free(hA); free(hB); free(hC);
    return 0;
}
```

---

## 6. Compile and Run

### 6.1 Compile

On Windows (from the x64 Native Tools Command Prompt):

```bat
nvcc -O2 -arch=sm_120 -o exp6.exe exp6.cu
```

On Linux / Colab:

```bash
nvcc -O2 -o exp6 exp6.cu
```



### 6.2 Run

```bat
exp6.exe                 :: default sizes: transpose N=4096, matmul N=1024
exp6.exe 2048 512        :: optional: transpose N=2048, matmul N=512
```

The program was run **three times**; the first run after loading the GPU can be slightly slower, so the repeated values are the more reliable ones.

> 📸 **Screenshot: three program runs**
<img width="1600" height="900" alt="WhatsApp Image 2026-11-20 at 10 04 42 AM" src="https://github.com/user-attachments/assets/a153a392-742b-4b16-9945-e230f3df6547" />


---

## 7. Results

All three runs printed `PASS` for every kernel, so the GPU results match the CPU reference.

### 7.1 Raw output (Run 1)

```
GPU: NVIDIA GeForce RTX 5060 Ti (compute capability 12.0)

=== PART A: Matrix transpose (N = 4096) ===
Uncoalesced :    0.837 ms |   160.4 GB/s | PASS
Coalesced   :    0.414 ms |   324.1 GB/s | PASS
Speedup (uncoalesced / coalesced) = 2.02x

=== PART B: Naive matrix multiplication (N = 1024) ===
Naive matmul:    1.385 ms |  1550.1 GFLOPS | PASS

=== SUMMARY ===
Transpose uncoalesced : 0.837 ms
Transpose coalesced   : 0.414 ms
Naive matmul          : 1.385 ms
```

### 7.2 Observation table (per run)

| Run | Uncoalesced (ms) | Uncoalesced (GB/s) | Coalesced (ms) | Coalesced (GB/s) | Speedup | Naive matmul (ms) | Matmul (GFLOPS) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.837 | 160.4 | 0.414 | 324.1 | 2.02× | 1.385 | 1550.1 |
| 2 | 0.803 | 167.1 | 0.415 | 323.6 | 1.94× | 1.385 | 1550.4 |
| 3 | 0.803 | 167.1 | 0.416 | 323.0 | 1.93× | 1.385 | 1550.5 |
| **Average** | **0.814** | **164.9** | **0.415** | **323.6** | **1.96×** | **1.385** | **1550.3** |

### 7.3 Observation table

| Kernel | Matrix size N | Time (ms) | Speedup | Bandwidth (GB/s) / GFLOPS | Result |
|---|:---:|:---:|:---:|:---:|:---:|
| Transpose: uncoalesced | 4096 | 0.814 | 1.0× (baseline) | 164.9 GB/s | PASS |
| Transpose: coalesced | 4096 | 0.415 | 1.96× | 323.6 GB/s | PASS |
| Naive matrix multiplication | 1024 | 1.385 | – | 1550.3 GFLOPS | PASS |

**Speedup of coalesced over uncoalesced transpose = 1.96×**

**GPU model used:** NVIDIA GeForce RTX 5060 Ti  **Compute capability:** 12.0

*(Values are averages of three runs. Run 1 was the slowest uncoalesced run, consistent with the manual's note that the first run can be slightly slower.)*

### 7.4 Sanity check of the formulas

For N = 4096 the matrix holds 4096 × 4096 × 4 B = 67,108,864 B. One read plus one write moves 134,217,728 B, so at 0.415 ms the coalesced kernel reaches 134,217,728 / (0.415 × 10⁶) ≈ **323.4 GB/s**, matching the printed value. For the matmul, 2 × 1024³ ≈ 2.147 × 10⁹ operations in 1.385 ms gives ≈ **1550 GFLOPS**, also matching.

---

## 8. Analysis and Discussion

**Coalesced vs. uncoalesced transpose.** Both kernels move the same data (134 MB per launch), yet the coalesced one is ~1.96× faster (164.9 → 323.6 GB/s). In the uncoalesced kernel each warp writes 32 floats that are `N` elements apart, so one warp-write splits into many memory transactions. In the coalesced kernel the shared-memory tile absorbs the transposition, so both the global read and the global write are contiguous. The `TILE + 1` padding keeps the column reads of the tile free of shared-memory bank conflicts.

**Speedup compared with the manual's sample.** The manual's illustrative T4 output shows 2.57×; here it is 1.96×. This is probably because the newer GPU's larger L2 cache merges some of the strided writes, which reduces the penalty of the uncoalesced kernel. This is an interpretation, not something the experiment measured.

**Bandwidth.** The coalesced transpose reaches ~324 GB/s, roughly 70% of the card's ~448 GB/s peak memory bandwidth (public specification; verify against the datasheet). The uncoalesced kernel reaches only about 37%.

**Naive matrix multiplication.** Each thread does N iterations with 2 global reads each, so the kernel issues about one global load per floating-point operation and has no data reuse. It reaches ~1550 GFLOPS, a small fraction (~6–7%) of the GPU's theoretical FP32 peak of roughly 23 TFLOPS (public specification), so it is memory-bound, not compute-bound. The figure is fairly high for a naive kernel because warps read `B` coalesced and `A` as a broadcast, and at N = 1024 each matrix (4 MB) is largely served from L2 cache. Tiling `A` and `B` in shared memory is the next optimisation.

**Timing method.** CUDA events measure time on the GPU itself; the warm-up launch and averaging over repeats remove start-up cost and noise. The run-to-run spread is small (0.803–0.837 ms uncoalesced, 0.414–0.416 ms coalesced), so the measurements are stable.

---

## 9. Conclusion

The experiment shows that the same amount of data movement can run at very different speeds depending on the access pattern. On the NVIDIA GeForce RTX 5060 Ti (compute capability 12.0), the coalesced shared-memory transpose of a 4096 × 4096 matrix took **0.415 ms (323.6 GB/s)** versus **0.814 ms (164.9 GB/s)** for the strided version, a **1.96× speedup**, with all results verified against the CPU reference. The naive matrix multiplication (N = 1024) ran in **1.385 ms (≈1550 GFLOPS)**, which is only a small fraction of the GPU's peak compute rate and confirms that a kernel with no explicit data reuse remains limited by global-memory traffic. Shared-memory tiling is the natural next optimisation.
