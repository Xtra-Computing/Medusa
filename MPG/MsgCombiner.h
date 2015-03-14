#ifndef MSGCOMBINER_H
#define MSGCOMBINER_H
#include "Configuration.h"
#include "../MultipleGPU/MultiGraphStorage.h"

#ifdef __LINUX__
#include <sys/time.h>
#include <unistd.h>
#endif

#include <time.h>


//in millisecond
void printTimestamp()
{

#ifdef __LINUX__
	struct timeval  tv;
	struct timezone tz;


	struct tm      *tm;
	long long         start;

	gettimeofday(&tv, &tz);

	start = tv.tv_sec * 1000000 + tv.tv_usec;

	printf("%lld\n",start);
#endif


}


void *combine_reuse(void *did)
{
	int gpu_id = (long) did;
	if(cudaSetDevice(gpu_id) != cudaSuccess)
        {
                printf("combiner thread set device error (%d)\n", gpu_id);
                exit(-1);
        }

	while(true)
	{
		pthread_mutex_lock(&MGLOBAL::com.common_mutex);
		pthread_cond_wait(&MGLOBAL::com.common_condition, &MGLOBAL::com.common_mutex);
		pthread_mutex_unlock(&MGLOBAL::com.common_mutex);


		cudppSegmentedScan(MGLOBAL::gpu_def[gpu_id].scanplan, MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.d_val, MGLOBAL::gpu_def[gpu_id].d_messageArray.d_val, MGLOBAL::gpu_def[gpu_id].d_edgeArray.d_incoming_msg_flag, MGLOBAL::gpu_def[gpu_id].d_messageArray.size);
        	MyCheckErrorMsg("after cudppSegmentedScan");


		cudaDeviceSynchronize();
        	MyCheckErrorMsg("after sync cudppSegmentedScan");
		
		pthread_mutex_lock(&MGLOBAL::com.individual_mutex[gpu_id]);
		pthread_cond_signal(&MGLOBAL::com.individual_condition[gpu_id]);
		pthread_mutex_unlock(&MGLOBAL::com.individual_mutex[gpu_id]);

	}
	return 0;
}

void *combine(void *did)
{

	int gpu_id = (long) did;

#ifdef VIS	

	cudppSegmentedScan(scanplan, GRAPH_STORAGE_CPU::alias_d_messageArrayBuf.d_val_x, GRAPH_STORAGE_CPU::alias_d_messageArray.d_val_x, GRAPH_STORAGE_CPU::alias_d_edgeArray.d_incoming_msg_flag, GRAPH_STORAGE_CPU::alias_d_messageArray.size);
	cudaThreadSynchronize();
	cutilCheckMsg("cudppSegmentedScan");

	cudppSegmentedScan(scanplan, GRAPH_STORAGE_CPU::alias_d_messageArrayBuf.d_val_y, GRAPH_STORAGE_CPU::alias_d_messageArray.d_val_y, GRAPH_STORAGE_CPU::alias_d_edgeArray.d_incoming_msg_flag, GRAPH_STORAGE_CPU::alias_d_messageArray.size);
	cudaThreadSynchronize();
	cutilCheckMsg("cudppSegmentedScan");


#else
	//need to use multi-threading to achieve multiple GPU parallel execution

	if(cudaSetDevice(gpu_id) != cudaSuccess)
	{
		printf("combiner thread set device error (%d)\n", gpu_id);
		exit(-1);
	}




	DBGPrintf("segment scan array size: %d\n",MGLOBAL::gpu_def[gpu_id].d_messageArray.size);


	
	
	//printf("GPU %d before segscan:",gpu_id);printTimestamp();
	
	cudppSegmentedScan(MGLOBAL::gpu_def[gpu_id].scanplan, MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.d_val, MGLOBAL::gpu_def[gpu_id].d_messageArray.d_val, MGLOBAL::gpu_def[gpu_id].d_edgeArray.d_incoming_msg_flag, MGLOBAL::gpu_def[gpu_id].d_messageArray.size);
	MyCheckErrorMsg("after cudppSegmentedScan");



	cudaDeviceSynchronize();
	MyCheckErrorMsg("after sync cudppSegmentedScan");
	


	return 0;

#endif

}

#endif
