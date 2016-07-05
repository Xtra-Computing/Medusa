//#include <cutil.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "../Algorithm/Configuration.h"

#ifdef VIS
#include <cuda_gl_interop.h>
//#include <cutil_inline_runtime.h>

#include <helper_functions.h>
#include <helper_timer.h>
#endif

#include "GraphStorage.h"


/**
* CPU side memory allocator
* @param [in] ptr predefined CPU pointer
* @param [in] size allocate size bytes memory
*/
void CPUMalloc(void **ptr, int size)
{
	(*ptr) = (void*)malloc(size);
	if((*ptr) == NULL)
	{
		printf("CPUMalloc Failed!\n");
		exit(-1);
	}
}

/**
* CPU side memory allocator
* @param [in] D_ptr predefined CPU pointer
* @param [in] size allocate size bytes memory
*/
void GPUMalloc(void **D_ptr, int size)
{
	CUDA_SAFE_CALL(cudaMalloc(D_ptr, size));
}


/**
* generate random number in the range of 0 ~ 2^32 - 1
* @note unsigned int type should be at least 32bits
*/

unsigned int rand32()
{
	unsigned int b8 = 0;
	unsigned int b16 = 0;
	unsigned int b24 = 0;
	unsigned int b32 = 0;
	b8 = rand()%256 + 1;
	b16 = rand()%256 + 1;
	b24 = rand()%256 + 1;
	b32 = rand()%256;
	unsigned r32 = b8 + (b16<<8) + (b24<<16) + (b32<<24);
	return r32;
}

/**
*
*
* @param []	
* @return
* @note	
*
*/
void InitCUDA()
{


	int count = 0;
	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device. Aborted!\n");
		exit(-1);
	}
#ifdef VIS
	cudaGLSetGLDevice(0);
	cudaGetDeviceProperties(&GRAPH_STORAGE_CPU::device_prop, 0);

#else
	cudaSetDevice(0);
	cudaGetDeviceProperties(&GRAPH_STORAGE_CPU::device_prop, 0);
#endif
	printf("CUDA initialized.\n");


// cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
}
