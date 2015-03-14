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
	//MVT *dbg = (MVT*)malloc(sizeof(MVT)*GRAPH_STORAGE_CPU::alias_d_edgeArray.size);
	//CUDA_SAFE_CALL(cudaMemcpy(dbg, GRAPH_STORAGE_CPU::alias_d_messageArray.d_val, sizeof(MVT)*GRAPH_STORAGE_CPU::alias_d_edgeArray.size, cudaMemcpyDeviceToHost));
	//for(int i = 0; i < GRAPH_STORAGE_CPU::alias_d_edgeArray.size; i++)
	//	printf("[%d]%f ",i, dbg[i]);
	int gpu_id = (long) did;
	//printf("GPU %d enter thread:",gpu_id);printTimestamp();

	//need to use multi-threading to achieve multiple GPU parallel execution

	medusaSetDevice(gpu_id);

	//debug: print message buffer

/*
	MVT *h_msg_buf = (MVT*)malloc(sizeof(MVT)*MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.size);
	cudaMemcpy(h_msg_buf, MGLOBAL::gpu_def[gpu_id].d_messageArray.d_val, sizeof(MVT)*MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.size, cudaMemcpyDeviceToHost);
	printf("print message buffer GPU %d:\n",gpu_id);
	MVT msg_sum = 0.0;
	printf("msg_buffer_size = %d\n",MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.size);
	for(int j = 0; j < MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.size; j ++)
	{
		printf("%f ",h_msg_buf[j]);
		msg_sum += h_msg_buf[j];
	}
	printf("sum:%f\n", msg_sum);
*/


	//debug: print segment scan index
	/*
	int *h_seg_flag = (int*)malloc(sizeof(int)*MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.size);
	cudaMemcpy(h_seg_flag, MGLOBAL::gpu_def[gpu_id].d_edgeArray.d_incoming_msg_flag, sizeof(int)*MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.size, cudaMemcpyDeviceToHost);
	printf("Print flags:\n");
	for(int j = 0; j < MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.size; j ++)
	printf("%d ",h_seg_flag[j]);
	printf("\n");
	*/
	
	//cutilCheckMsg("before cudppSegmentedScan");


	//printf("segment scan array size: %d\n",MGLOBAL::gpu_def[gpu_id].d_messageArray.size);


	/*
	unsigned int timer;
	float duration;
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	*/
	
	//printf("GPU %d before segscan:",gpu_id);printTimestamp();
	
	cudppSegmentedScan(MGLOBAL::gpu_def[gpu_id].scanplan, MGLOBAL::gpu_def[gpu_id].d_messageArrayBuf.d_val, MGLOBAL::gpu_def[gpu_id].d_messageArray.d_val, MGLOBAL::gpu_def[gpu_id].d_edgeArray.d_incoming_msg_flag, MGLOBAL::gpu_def[gpu_id].d_messageArray.size);
	MyCheckErrorMsg("after cudppSegmentedScan");

	//printf("GPU %d after segscan:",gpu_id);printTimestamp();

	//debug: print combined buffer
	/*
	msg_sum = 0.0;
	cudaMemcpy(h_msg_buf, MGLOBAL::gpu_def[i].d_messageArrayBuf.d_val, sizeof(MVT)*MGLOBAL::gpu_def[i].d_messageArrayBuf.size, cudaMemcpyDeviceToHost);
	printf("print combined message buffer:\n");

	printf("msg_buffer_size = %d\n",MGLOBAL::gpu_def[i].d_messageArrayBuf.size);
	for(int j = 0; j < MGLOBAL::gpu_def[i].d_messageArrayBuf.size; j ++)
	msg_sum += h_msg_buf[j];

	printf("sum:%f\n", msg_sum);

	medusaSetDevice(i);
	cudaDeviceSynchronize();
	cutilCheckMsg("early sync cudppSegmentedScan");
	*/

	cudaDeviceSynchronize();
	
	//printf("GPU %d sync segscan:",gpu_id);printTimestamp();

	MyCheckErrorMsg("sync cudppSegmentedScan");

/*
	cutStopTimer(timer);
	duration = cutGetTimerValue(timer);
	printf("scan time: %f ms\n", duration);
*/	
	
	return 0;


}

#endif
