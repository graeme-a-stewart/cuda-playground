// Trivial array add example
// This is almost like "hello, world" in CUDA :-)

#include <iostream>

const size_t array_size = 1024;

// Good macro for making sure we know where things went wrong...
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
    file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

// __global__ means this is a piece of device code
__global__
void add_kernel(float* c, int n_el, const float* a, const float* b) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n_el)
    return;

  c[index] = a[index] + b[index];
}

int main() {
  float host_array_a[array_size];
  float host_array_b[array_size];
  float host_array_c[array_size];

  for (size_t i=0; i<array_size; ++i) {
    host_array_a[i] = i;
    host_array_b[i] = 8*i;
  }

  // Pointers for device memory blocks
  // Note that cudaMalloc wants a pointer to the pointer
  float *device_array_a, *device_array_b, *device_array_c;
  checkCudaErrors(cudaMalloc(&device_array_a, sizeof(host_array_a)));
  checkCudaErrors(cudaMalloc(&device_array_b, sizeof(host_array_b)));
  checkCudaErrors(cudaMalloc(&device_array_c, sizeof(host_array_c)));

  checkCudaErrors(cudaMemcpy(device_array_a, host_array_a, sizeof(host_array_a), 
    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(device_array_b, host_array_b, sizeof(host_array_b), 
    cudaMemcpyHostToDevice));

  // Define block size and number, then launch the kernel
  const int block_size = 128;
  int n_blocks = (array_size + block_size - 1) / block_size;
  add_kernel<<<block_size, n_blocks>>>(device_array_c, array_size,
    device_array_a, device_array_b);

  // Copy back
  checkCudaErrors(cudaMemcpy(host_array_c, device_array_c, sizeof(host_array_c),
    cudaMemcpyDeviceToHost));

  for (size_t i=0; i<array_size; ++i) {
    std::cout << host_array_a[i] << " + " << host_array_b[i] << " = " <<
      host_array_c[i] << std::endl;
  }

  return 0;
}