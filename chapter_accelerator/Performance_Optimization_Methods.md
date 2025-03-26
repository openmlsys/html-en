# Performance Optimization Methods

Hardware accelerators boast intricate computational and memory
architectures. To maximize their performance, developers frequently need
to grasp a variety of performance optimization methods. Common methods
encompass enhancing arithmetic intensity, capitalizing effectively on
shared memory, optimizing the memory load/store pipeline, among others.
The subsequent sections will elucidate these methods through practical
programming examples, all aimed towards a singular objective:
accelerating an FP32 GEMM program.

## Implementing General Matrix Multiplication

Code `lst:cpu` shows a reference implementation of GEMM in C++.

**lst:cpu**
```cpp
float A[M][K];
float B[K][N];
float C[M][N];
float alpha, beta;

for (unsigned m = 0; m < M; ++m) {
    for (unsigned n = 0; n < N; ++n) {
        float c = 0;
        for (unsigned k = 0; k < K; ++k) {
            c += A[m][k] * B[k][n];
        }
        C[m][n] = alpha * c + beta * C[m][n];
    }
}
```

ach element in matrix $C$ is independently computed, and numerous GPU
threads can be launched to compute the corresponding elements in matrix
$C$ in parallel. The GPU kernel function is shown in
Code `lst:gpu`.

**lst:gpu**
```cpp
__global__ void gemmKernel(const float * A,
const float * B, float * C,
float alpha, float beta, unsigned M, unsigned N,
unsigned K) {
    unsigned int m = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int n = threadIdx.y + blockDim.y * blockIdx.y;
    if (m >= M || n >= N)
    return;
    float c = 0;
    for (unsigned k = 0; k < K; ++k) {
        c += A[m * K + k] * B[k * N + n];
    }
    c = c * alpha;
    float result = c;
    if (beta != 0) {
        result = result + C[m * N + n] * beta;
    }
    C[m * N + n] = result;
}
```

Figure :numref:`cuda_naive_gemm` shows the layout of the implementation.
Each element in matrix $C$ is computed by one thread. The row index $m$
and column index $n$ of the element in matrix $C$ corresponding to the
thread are computed in lines 5 and 6 of the GPU kernel. Then, in lines 9
to 11, the thread loads the row vector in matrix $A$ according to the
row index and the column vector in matrix $B$ according to the column
index, computes the vector inner product. The thread also stores the
result back to $C$ matrix in line 17.

![Simple implementation ofGEMM](../img/ch06/practise/naive.png)
:label:`cuda_naive_gemm`

The method of launching the kernel function is shown in
Code `lst:launch`.

**lst:launch**
```cpp
void gemmNaive(const float *A, const float *B, float *C,
float alpha, float beta, unsigned M,
unsigned N, unsigned K) {
    dim3 block(16, 16);
    dim3 grid((M - 1) / block.x + 1, (N - 1) / block.y + 1);
    
    gemmKernel<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);
}
```

Each thread block processes $16\times16$ elements in matrix $C$.
Therefore, $(M - 1) / 16 + 1 \times (N - 1) / 16 + 1$ thread blocks are
used to compute the entire matrix $C$.

Eigen is used to generate data and compute the GEMM result on the CPU.
In addition, error computing and time profiling code are implemented for
the GPU computing result. For details, see
[first_attempt.cu](https://github.com/openmlsys/openmlsys-cuda/blob/main/first_attempt.cu).
After the program is compiled and executed, output results are as
follows:

```
FP32 peak throughput 29767.680 GFLOPS
Average Throughput: 185.313 GFLOPS
```

A significant gap exists between the performance that can be achieved by
the current code and the peak device performance. In an entire computing
process, the process with the highest computing density is matrix
multiplication $A\times B$. Its time complexity is $O(M*N*K)$, whereas
that time complexity of the entire computing process is
$O(M*N*K+2*M*N)$. Therefore, optimizing matrix multiplication is key to
improving performance.
