#ifndef PROCLUS_GPU_GPU_UTIL_H
#define PROCLUS_GPU_GPU_UTIL_H
#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// --- CUDA device header only when compiled by NVCC ---
// (prevents MSVC from pulling in device-only headers like <curand_kernel.h>)
#if defined(__CUDACC__)
  #include <curand_kernel.h>
  using curandState_t = curandState;     // real type under NVCC
#else
  struct curandState;                    // forward declaration for MSVC
  using curandState_t = curandState;     // pointer types compile fine on host
#endif

// ------------------------------------------------------------------
// Declarations (unchanged API surface; only guarded where necessary)
// ------------------------------------------------------------------

// print helpers
void print_array_gpu(int *d_X, int n);
void print_array_gpu(float *d_X, int n);
void print_array_gpu(bool *d_X,   int n);
void print_array_gpu(bool *d_X,   int n, int m);

// shuffling / sampling
int*  gpu_shuffle(int *h_indices, int n);
void  gpu_shuffle_v2(int *d_a, int *h_indices, int n);
void  gpu_shuffle_v3(int *d_in, int n, curandState_t *d_state);
void  gpu_random_sample(int *d_in, int k, int n, curandState_t *d_state);
void  gpu_random_sample_locked(int *d_in, int k, int n, curandState_t *d_state, int *d_lock);
void  gpu_not_random_sample_locked(int *d_in, int k, int n, int *d_state, int *d_lock);

// transfers / gathers
float* copy_to_flatten_device(at::Tensor h_mem, int height, int width);
float* gpu_gather_2d(float *d_source, int *d_indices, int height, int width);
void   gpu_gather_1d(int *d_result, int *d_source, int *d_indices, int length);

// setters (host-callable wrappers)
void set(int *x, int i, int value);
void set(int *x, int *idx, int i, int value);
void set(float *x, int i, float value);

// kernels (visible only to NVCC—MSVC never sees __global__)
#if defined(__CUDACC__)
__global__ void init_seed(curandState_t *state, int seed);
__global__ void set_all(float *d_X, float value, int n);
__global__ void set_all(int   *d_X, int   value, int n);
__global__ void set_all(bool  *d_X, bool  value, int n);
#endif

// misc GPU helpers
void  gpu_clone(int *d_to, int *d_from, int size);

// If this is device-only, consider moving to a .cu or guarding with __CUDACC__.
// If it's a host-side algorithm, leave it as-is.
void  inclusive_scan(int *source, int *result, int n);

// device allocators
int*   device_allocate_int(int n);
float* device_allocate_float(int n);
bool*  device_allocate_bool(int n);

int*   device_allocate_int_zero(int n);
float* device_allocate_float_zero(int n);
bool*  device_allocate_bool_zero(int n);

// accounting
int   get_total_allocation_count();
void  add_total_allocation_count(int n);
void  reset_total_allocation_count();

#endif // PROCLUS_GPU_GPU_UTIL_H