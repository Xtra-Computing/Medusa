#ifndef COMBINER_H
#define COMBINER_H

#include <cudpp.h>
#include <pthread.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <cuda_runtime.h>

#include "../Compatibility/Compatability.h"

/**
*
*
* @param []
* @return
* @note	
*
*/


struct Medusa_Combiner
{

	pthread_t *combine_thread;
	void init(CUDPPDatatype dt, CUDPPOperator op, int gpu_id);
	//void combine(void *gpu_id);
	void combineAllDevice();

	pthread_mutex_t common_mutex;
	pthread_mutex_t *individual_mutex;
	pthread_cond_t  common_condition;
	pthread_cond_t *individual_condition;


};


#endif
