#include <cutil_inline.h>


template <class T>
__global__ void gather_kernel(T *dst, T *src, int *index, int thread_num, int element_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(; tid < element_num; tid += thread_num)
	{
		dst[tid] = src[index[tid]];
	}
}


template <class T>
__global__ void scatter_kernel(T *dst, T *src, int *index, int thread_num, int element_num)
{
	//printf("enter scatter\n");
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(; tid < element_num; tid += thread_num)
	{
		dst[index[tid]] = src[tid];
	}
}
