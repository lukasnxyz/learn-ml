#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256  // number of rows in A and C
#define K 512  // number of columns in A and rows in B
#define N 256  // number of columns in B and C
#define BLOCK_SIZE 32

// naive matmul ofc

// m: rows in A and C, k: cols in A and rows in B, n: cols in B and C
void matmul_cpu(float *A, float *B, float *C, uint32_t m, uint32_t k, uint32_t n) {
  for (uint32_t i = 0; i < m; ++i) { // each row in C
    for (uint32_t j = 0; j < n; ++j) { // each col in C
      float sum = 0.0f;

      for (uint32_t l = 0; l < k; ++l) // each col in A, row in B
        sum += A[i * k + l] * B[l * n + j]; // mul each at the current row and col

      C[i * n + j] = sum; // set sum in output matrix
    }
  }
}

__global__ void matmul_gpu(float *A, float *B, float *C, uint32_t m, uint32_t k, uint32_t n) {
  const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float sum = 0.0f;

    for (uint32_t l = 0; l < k; ++l)
      sum += A[row * k + l] * B[l * n + col];

    C[row * n + col] = sum;
  }
}

void init_matrix(float *mat, uint32_t rows, uint32_t cols) {
  for (uint32_t i = 0; i < rows * cols; ++i)
    mat[i] = (float)rand() / RAND_MAX;
}

double get_time() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
  float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
  float *d_A, *d_B, *d_C;
  const uint32_t size_A = M * K * sizeof(float);
  const uint32_t size_B = K * N * sizeof(float);
  const uint32_t size_C = M * N * sizeof(float);

  h_A = (float *)malloc(size_A);
  h_B = (float *)malloc(size_B);
  h_C_cpu = (float *)malloc(size_C);
  h_C_gpu = (float *)malloc(size_C);

  srand(time(NULL));
  init_matrix(h_A, M, K);
  init_matrix(h_B, K, N);

  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  const dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

  // ------

  const uint32_t warm_up_runs = 100;
  const uint32_t benchmark_runs = 500;

  printf("Performing warm-up runs...\n");
  for (uint32_t i = 0; i < warm_up_runs; ++i) {
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
  }

  printf("Benchmarking CPU implementation...\n");
  double cpu_total_time = 0.0;
  for (uint32_t i = 0; i < benchmark_runs; ++i) {
    const double start_time = get_time();

    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);

    const double end_time = get_time();
    cpu_total_time += end_time - start_time;
  }
  const double cpu_avg_time = cpu_total_time / benchmark_runs;

  printf("Benchmarking GPU implementation...\n");
  double gpu_total_time = 0.0;
  for (uint32_t i = 0; i < benchmark_runs; ++i) {
    const double start_time = get_time();

    matmul_gpu<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    const double end_time = get_time();
    gpu_total_time += end_time - start_time;
  }
  const double gpu_avg_time = gpu_total_time / benchmark_runs;

  printf("CPU average time: %f microseconds\n", (cpu_avg_time * 1e6f));
  printf("GPU average time: %f microseconds\n", (gpu_avg_time * 1e6f));
  printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

  // ------

  free(h_A);
  free(h_B);
  free(h_C_cpu);
  free(h_C_gpu);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}
