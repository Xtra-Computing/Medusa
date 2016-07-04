#ifndef INITMESSAGE_H
#define INITMESSAGE_H

__global__ void MsgArrayInit(Message init_val, int total_thread_count)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for(; tid < GRAPH_STORAGE_GPU::d_messageArray.size; tid += total_thread_count)
		GRAPH_STORAGE_GPU::d_messageArray.d_val[tid] = init_val.val;
}

#endif