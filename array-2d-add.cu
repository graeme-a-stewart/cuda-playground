// Array add example, 2D

#include <iostream>
#include <iomanip>

const size_t array_size_x = 32;
const size_t array_size_y = 16;

// Good macro for making sure we know where things went wrong...
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error: " << cudaGetErrorString(result) << " (" <<
    static_cast<unsigned int>(result) << ") at " <<
    file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

// __global__ means this is a piece of device code
__global__
void add_kernel(float* c, int max_x, int max_y, const float* a, const float* b) {
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;
  printf("hello, world! Block (%d, %d); Thread (%d, %d); Index (%d, %d)\n",
    blockIdx.x, blockIdx.y,
    threadIdx.x, threadIdx.y,
    index_x, index_y);
  if (index_x >= max_x || index_y >= max_y)
    return;

  c[index_x + index_y*max_x] = a[index_x + index_y*max_x] + b[index_x + index_y*max_x];
}

int main() {
  size_t array_size = array_size_x * array_size_y;

  float host_array_a[array_size];
  float host_array_b[array_size];
  float host_array_c[array_size];

  for (size_t m=0; m<array_size_x; ++m) {
    for (size_t n=0; n<array_size_y; ++n) {
      host_array_a[m + n*array_size_x] = m*n;
      host_array_b[m + n*array_size_x] = m+n;
    }
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
  dim3 block_dim(8, 8);
  dim3 grid_dim(4, 2);
  add_kernel<<<block_dim, grid_dim>>>(device_array_c, array_size_x, array_size_y,
    device_array_a, device_array_b);

  // Copy back
  checkCudaErrors(cudaMemcpy(host_array_c, device_array_c, sizeof(host_array_c),
    cudaMemcpyDeviceToHost));

  for (size_t m=0; m<array_size_x; ++m) {
    for (size_t n=0; n<array_size_y; ++n) {
      std::cout << std::setw(4) << host_array_c[m + n*array_size_x] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}