#include <iostream>

// A simple CUDA kernel: adds two vectors element-wise
__global__ void addVectors(const int *a, const int *b, int *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    c[idx] = a[idx] + b[idx];
}

int main() {
  int n = 10; // size of vectors
  int bytes = n * sizeof(int);

  int h_a[10], h_b[10], h_c[10];
  for (int i = 0; i < n; i++) {
    h_a[i] = i;
    h_b[i] = i * 2;
  }

  printf("before element wise addition:\n");
  printf("a: ");
  for (int i = 0; i < n; i++) printf("%d ", h_a[i]);
  printf("\nb: ");
  for (int i = 0; i < n; i++) printf("%d ", h_b[i]);
  printf("\n");

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from host → device
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  // Launch kernel with 1 block of 256 threads (enough for n=10)
  addVectors<<<1, 256>>>(d_a, d_b, d_c, n);

  // Copy result back to host
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // Print results
  printf("result c: ");
  for (int i = 0; i < n; i++) printf("%d ", h_c[i]);
  printf("\n");

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
