#ifndef APIWRAPPER_H
#define APIWRAPPER_H


#include "../Algorithm/DeviceDataStructure.h"
#include "../Algorithm/Configuration.h"

template<class O>
__global__ void vertexProcWrapper(int super_step, O op, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for(; tid < GRAPH_STORAGE_GPU::d_vertexArray.size; tid += total_thread_num)
	{
		D_Vertex vertex(tid);
		op(vertex, super_step);
	}
}




template<class O>
__global__ void OOvertexProcWrapper(int super_step, O op, int offset, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for(; tid < offset; tid += total_thread_num)
	{
		D_Vertex vertex(tid);
		op(vertex, super_step);
	}
}


template<class O>
__global__ void edgeProcWrapper(int super_step, O op, int total_thread_num)
{
	int tid = blockDim.x*blockIdx.x + threadIdx.x;

	for(; tid < GRAPH_STORAGE_GPU::d_edgeArray.size; tid += total_thread_num)
	{
		D_Edge edge(tid);
		op(edge, super_step);
	}
}

/*
 *	Queue version of edgeProcWrapper
 *	Work on edges (either edgeArray or edge queue), generate edge queu
 *	Code for enqueue operation is added before and after op operator.
 *	The op operator will call SetActive(int vertexID)
 */

#if defined MBFS
#define SetActive(vertexID)\
{\
	activeVertexID = vertexID;\
}\


//The number of threads should be not less than edgeQueueAlphaSize
#if defined(WARPBASED) or defined (SCANBASED)
#define WARPS 8
#define WSIZE 32
//#define ATOMIC
__global__ void EdgeQueue2EdgeQueue(int super_step, int *edgeQueuePtr, int *edgeQueueAlpha, int edgeQueueAlphaSize, int *edgeQueueBeta)
{
	int tid = blockDim.x*(blockIdx.x*gridDim.y + blockIdx.y) + threadIdx.x;
#if defined WARPBASED
	volatile __shared__  int comm[WARPS][3];
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int activeVertexID = -1;
#endif
	if(tid < edgeQueueAlphaSize)
	{
//----------------------generated code------------------------------------

		int dstVertexID = edgeQueueAlpha[tid];
		D_Vertex dst_V(dstVertexID);
#if defined ATOMIC
		int dstLevel = atomicCAS(&GRAPH_STORAGE_GPU::d_vertexArray.d_level[dstVertexID], MVT_Init_Value, super_step + 1);
		if(dstLevel == MVT_Init_Value)
		{
			SetActive(dstVertexID);
		}
#else
		if(dst_V.get_level() == MVT_Init_Value)
		{
			dst_V.set_level(super_step + 1);
			SetActive(dstVertexID);
		}
#endif



//--------------------end of generated code--------------------------------
	}
#if defined WARPBASED
	//enqueue the edges
	int r = 0, r_end = 0;
	if(activeVertexID != -1)
	{
		r = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[activeVertexID];
		r_end = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[activeVertexID + 1];
	}
	
	//printf("%d\n", activeVertexID);
    while(__any(r_end - r > 0))
	{
		//vie for control of warp
		if(r_end - r)
		{
			comm[warp_id][0] = lane_id;
		}

		// winner describes adjlist
		if (comm[warp_id][0] == lane_id)
		{
			comm[warp_id][1] = r;
			comm[warp_id][2] = r_end;
			//get enqueue offset of this edgelist
			comm[warp_id][0] = atomicAdd(edgeQueuePtr, r_end - r);
			r_end = r;
		}

		//strip-mine winner's adjlist
		int r_gather = comm[warp_id][1] + lane_id;
		int r_gather_end = comm[warp_id][2];
		int r_scatter = comm[warp_id][0] + lane_id;
		while(r_gather < r_gather_end)
		{
			edgeQueueBeta[r_scatter] = GRAPH_STORAGE_GPU::d_edgeArray.d_dstVertexID[r_gather];
			r_gather += WSIZE;
			r_scatter += WSIZE;
		}
	}
#endif
}
#endif //defined(WARPBASED) or defined (SCANBASED)
#endif //defined MBFS

/**
*
* Edge processor wrapper for original and core
* @param   - 
* @return	
* @note	
*
*/
template<class O>
__global__ void OOEdgeProcWrapper(int super_step, O op, int offset, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	for(; tid < offset; tid += total_thread_num)
	{
		D_Edge edge(tid);
		op(edge, super_step);
	}
}

/**
*
* Edge processor wrapper for replica
* @param   - 
* @return	
* @note	
*
*/
template<class O>
__global__ void REdgeProcWrapper(int super_step, O op, int offset, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;
	
	for(;tid < GRAPH_STORAGE_GPU::d_edgeArray.size; tid += total_thread_num)
	{
		D_Edge edge(tid);
		op(edge, super_step);
	}
}



template<class O>
__global__ void edgeProcWrapperHT(int super_step, O op, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for( ; tid < GRAPH_STORAGE_GPU::d_edgeArray.size; tid += total_thread_num)
	{
		D_Edge edge(tid);
		op(edge, super_step);
	}
}


template<class O>
__global__ void edgeListProcWrapper(int super_step, O op, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	for(; tid < GRAPH_STORAGE_GPU::d_vertexArray.size; tid += total_thread_num)
	{
		EdgeList edgeList(tid);
		D_Vertex vertex(tid);
		op(vertex, edgeList, super_step);
	}
}

template<class O>
__global__ void messageProcWrapper(int super_step, O op, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for(; tid < GRAPH_STORAGE_GPU::d_edgeArray.size; tid += total_thread_num)
	{
		D_Message message(tid);
		op(message, super_step);
	}
}


#endif
