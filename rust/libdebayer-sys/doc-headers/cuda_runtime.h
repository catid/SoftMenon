// Declaration-only subset for docs.rs; this is not a CUDA implementation.
#ifndef LIBDEBAYER_DOC_CUDA_RUNTIME_H
#define LIBDEBAYER_DOC_CUDA_RUNTIME_H
#include <stddef.h>
typedef struct CUstream_st* cudaStream_t;
enum cudaError { cudaSuccess = 0 };
typedef enum cudaError cudaError_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2 };
cudaError_t cudaMallocPitch(void**, size_t*, size_t, size_t);
cudaError_t cudaMemset2D(void*, size_t, int, size_t, size_t);
cudaError_t cudaMemcpy2DAsync(void*, size_t, const void*, size_t, size_t, size_t, enum cudaMemcpyKind, cudaStream_t);
cudaError_t cudaStreamCreate(cudaStream_t*);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaFree(void*);
cudaError_t cudaGetLastError(void);
#endif
