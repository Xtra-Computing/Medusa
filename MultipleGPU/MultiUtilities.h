
#ifndef MULTIUTILITIES_H
#define MULTIUTILITIES_H



#include <cutil.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "../Algorithm/Configuration.h"
#ifdef VIS
#include <cuda_gl_interop.h>
#endif
#include <cutil_inline_runtime.h>
#include "../Algorithm/Configuration.h"
#include "../MultipleGPU/MultiGraphStorage.h"


void InitMultiGPU(int num_gpu_to_use)
{
	int device_num = -1;
	cudaGetDeviceCount(&device_num);
	if(device_num < MGLOBAL::num_gpu_to_use)
	{
		printf("!!!!This system only has %d GPUs, less than the specified number (%d)\n", device_num, MGLOBAL::num_gpu_to_use);
		//exit(-1);
	}
	MGLOBAL::gpu_def = (GPUDef *)malloc(sizeof(GPUDef)*MGLOBAL::num_gpu_to_use);
	for (int i=0; i < MGLOBAL::num_gpu_to_use; i++) {
		cutilSafeCall(cudaGetDeviceProperties(&MGLOBAL::gpu_def[i].device_prop, i));
		printf("> GPU %d  %s has asyncEngineCount %d %d\n", i, MGLOBAL::gpu_def[i].device_prop.name, MGLOBAL::gpu_def[i].device_prop.asyncEngineCount, MGLOBAL::gpu_def[i].device_prop.multiProcessorCount);
		printf("");
		if(MGLOBAL::gpu_def[i].device_prop.asyncEngineCount < 1)
		{
			printf("All GPUs must be capable of overlapping\n");
			//exit(-1);
		}
	}

}

//Need multi-threading? Not sure...

void ResetToExecute()
{
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		cudaMemcpyToSymbol((GRAPH_STORAGE_GPU::d_toExecute),&(MGLOBAL::toExecute) , sizeof(MGLOBAL::toExecute),0,cudaMemcpyHostToDevice);
	}
}

bool RetriveToExecute()
{
	bool to_execute = false;
	bool exe_temp;
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		cudaMemcpyFromSymbol(&exe_temp, (GRAPH_STORAGE_GPU::d_toExecute), sizeof(GRAPH_STORAGE_GPU::d_toExecute),0,cudaMemcpyDeviceToHost);
		//printf("C:GPU %d %d\n", i, int(exe_temp));
		to_execute = (to_execute || exe_temp);
	}
	return to_execute;
}

//return when update_replica_thread finishes
void syncEdgeOperator()
{
	int n;
	if ( n = pthread_join( MGLOBAL::replica_update_thread, NULL ) ) {
		fprintf( stderr, "pthread_join: %s\n", strerror( n ) );
		exit( 1 );
	}

}


#endif
