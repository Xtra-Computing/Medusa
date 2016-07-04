/****************************************************
* @file 
* @brief 
* @version
* @author Zhong Jianlong(http://www.jlzhong.com)
* @date 2011/01/06
* Copyleft for non-commercial use only. No warranty.
****************************************************/
#include "Combiner.h"
#include <helper_timer.h>

#include <cudpp.h>
#include "../MultipleGPU/MultiGraphStorage.h"
#include "../Algorithm/Configuration.h"


#include "../Algorithm/MsgCombiner.h"


//init for one GPU only, before calling this function the right GPU must be set by calling cudaSetDevice
void Medusa_Combiner::init(CUDPPDatatype dt, CUDPPOperator op, int gpu_id)
{
	
	
#ifdef VIS
	//visualization only uses the 0th GPU
	MGLOBAL::gpu_def[0].config.op = op;
	MGLOBAL::gpu_def[0].config.datatype = dt;
	MGLOBAL::gpu_def[0].config.algorithm = CUDPP_SEGMENTED_SCAN;
	MGLOBAL::gpu_def[0].config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;


	MGLOBAL::gpu_def[0].scanplan = 0;
	CUDPPResult result = cudppPlan(&MGLOBAL::gpu_def[0].scanplan, MGLOBAL::gpu_def[0].config, MGLOBAL::gpu_def[0].d_edgeArray.size, 1, 0); 
#else
	MGLOBAL::gpu_def[gpu_id].config.op = op;
	MGLOBAL::gpu_def[gpu_id].config.datatype = dt;
	MGLOBAL::gpu_def[gpu_id].config.algorithm = CUDPP_SEGMENTED_SCAN;
	MGLOBAL::gpu_def[gpu_id].config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;


#ifdef CUDPP_2_0
	MGLOBAL::gpu_def[gpu_id].cudpp_result = cudppCreate(&MGLOBAL::gpu_def[gpu_id].theCudpp);
	if(MGLOBAL::gpu_def[gpu_id].cudpp_result != CUDPP_SUCCESS)
	{
        	printf("Error initializing CUDPP Library\n");
        	exit(-1);
    	}
	MGLOBAL::gpu_def[gpu_id].cudpp_result = cudppPlan(MGLOBAL::gpu_def[gpu_id].theCudpp, &MGLOBAL::gpu_def[gpu_id].scanplan, MGLOBAL::gpu_def[gpu_id].config, MGLOBAL::gpu_def[gpu_id].d_messageArray.size, 1, 0);
#else
	MGLOBAL::gpu_def[gpu_id].scanplan = 0;
	CUDPPResult result = cudppPlan(&MGLOBAL::gpu_def[gpu_id].scanplan, MGLOBAL::gpu_def[gpu_id].config, MGLOBAL::gpu_def[gpu_id].d_messageArray.size, 1, 0); 
#endif
	//bug: the shared data structures should only be initialized once
	if(gpu_id == 0)
	{
		printf("init combiner\n");
		combine_thread = (pthread_t *)malloc(sizeof(pthread_t)*MGLOBAL::num_gpu_to_use);
		pthread_mutex_init(&common_mutex, NULL);
		pthread_cond_init(&common_condition, NULL);
		individual_condition = (pthread_cond_t *)malloc(sizeof(pthread_cond_t)*MGLOBAL::num_gpu_to_use);
		individual_mutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t)*MGLOBAL::num_gpu_to_use);
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			pthread_cond_init(&individual_condition[i], NULL);
			pthread_mutex_init(&individual_mutex[i], NULL);
		}
		for (int pi = 0; pi < MGLOBAL::num_gpu_to_use; pi ++)
        	        if(pthread_create(&combine_thread[pi], NULL, combine_reuse, (void*) pi))
			{
				printf("create thread failed\n");
				exit(-1);
			}
	}


#endif
	//printf("init combiner size = %d\n",GRAPH_STORAGE_CPU::alias_d_messageArray.size);
}


void Medusa_Combiner::combineAllDevice()
{
	int pi = 0;
	StopWatchInterface *timer = NULL;
	float duration;
	cutCreateTimer(&timer);
	cutResetTimer(timer);
	//printf("Before combiner timer:");printTimestamp();
	cutStartTimer(timer);

	if(MGLOBAL::num_gpu_to_use == 1)
	{
		combine((void*)pi);
		cutStopTimer(timer);
		//printf("After combiner timer:");printTimestamp();

		//duration = cutGetTimerValue(timer);
		//printf("combiner time: %f ms\n", duration);
		//printf("combiner time: %f ms\n", duration);
		return;
	}


/*
	for (; pi < MGLOBAL::num_gpu_to_use; pi ++)
		pthread_create(&combine_thread[pi], NULL, combine, (void*) pi);


	for(pi = MGLOBAL::num_gpu_to_use - 1; pi >= 0; pi --)
		pthread_join( combine_thread[pi], NULL );
*/


	for (pi = 0; pi < MGLOBAL::num_gpu_to_use; pi ++)
        {
                pthread_mutex_lock( &individual_mutex[pi] );
	}

	pthread_mutex_lock(&common_mutex );
	pthread_cond_broadcast(&common_condition);
	pthread_mutex_unlock(&common_mutex );

	for (pi = 0; pi < MGLOBAL::num_gpu_to_use; pi ++)
	{
		pthread_cond_wait(&individual_condition[pi], &individual_mutex[pi]);
		pthread_mutex_unlock( &individual_mutex[pi] );
	}
/*
	for(; pi < MGLOBAL::num_gpu_to_use; pi ++)
	{
		combine((void*)pi);
	}
*/
	cutStopTimer(timer);
	//printf("After combiner timer:");printTimestamp();

	duration = cutGetTimerValue(timer);
	DBGPrintf("combiner time: %f ms\n", duration);


}
