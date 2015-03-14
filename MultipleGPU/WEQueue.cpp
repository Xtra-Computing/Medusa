#include <cutil_inline.h>
#include "WEQueue.h"


/**********************************************************
 * Queue related functions.
 * Be careful to select the desired device before calling.
 *
 *********************************************************/
void SetEdgeQueuePtr( int _ptrValue)
{
	cudaMemcpy(MGLOBAL::gpu_def[MGLOBAL::current_device].edgeQueuePtr, &_ptrValue, sizeof(int), cudaMemcpyHostToDevice);
}

int GetEdgeQueuePtr()
{
	 int temp;
	    cudaMemcpy(&temp, MGLOBAL::gpu_def[MGLOBAL::current_device].edgeQueuePtr, sizeof(int), cudaMemcpyDeviceToHost);
	return temp;
}

void SetVertexQueuePtr( int _ptrValue)
{
	cudaMemcpy(MGLOBAL::gpu_def[MGLOBAL::current_device].vertexQueuePtr, &_ptrValue, sizeof(int), cudaMemcpyHostToDevice);
}

 int GetVertexQueuePtr()
{
	int temp;
    cudaMemcpy(&temp, MGLOBAL::gpu_def[MGLOBAL::current_device].vertexQueuePtr, sizeof(int), cudaMemcpyDeviceToHost);
	return temp;
}


void SwapEdgeQueue()
{
	int *temp;
	temp = MGLOBAL::gpu_def[MGLOBAL::current_device].edgeQueueAlpha;
	MGLOBAL::gpu_def[MGLOBAL::current_device].edgeQueueAlpha = MGLOBAL::gpu_def[MGLOBAL::current_device].edgeQueueBeta;
	MGLOBAL::gpu_def[MGLOBAL::current_device].edgeQueueBeta = temp;
}


void SwapVertexQueue()
{
	int *temp;
	temp = MGLOBAL::gpu_def[MGLOBAL::current_device].vertexQueueAlpha;
	MGLOBAL::gpu_def[MGLOBAL::current_device].vertexQueueAlpha = MGLOBAL::gpu_def[MGLOBAL::current_device].vertexQueueBeta;
	MGLOBAL::gpu_def[MGLOBAL::current_device].vertexQueueBeta = temp;
}

