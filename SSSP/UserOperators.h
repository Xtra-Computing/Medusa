
#ifndef USEROPERATORS_H
#define USEROPERATORS_H
#include "../MedusaRT/SystemLibGPU.h"
#include "Configuration.h"
#include <float.h>
//---------------------------------------------------------------------------------------------------------//
/* edge_list proc version */
#ifdef AA
__global__ void ssspEdgeProcAA(int super_step)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	//if(tid == 0)
	//printf("enter kernel %d\n",GRAPH_STORAGE_GPU::d_vertexArray.size);
	if(tid >= GRAPH_STORAGE_GPU::d_vertexArray.size)
		return;
	if(GRAPH_STORAGE_GPU::d_vertexArray.d_updated[tid])
	{

		int start_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid];
		int end_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid + 1];
		MVT distance = GRAPH_STORAGE_GPU::d_vertexArray.d_distance[tid];
		for(; start_index < end_index; start_index ++)
		{

			GRAPH_STORAGE_GPU::d_messageArray.d_val[GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[start_index]] = distance + GRAPH_STORAGE_GPU::d_edgeArray.d_weight[start_index];
		}
	}
}
#endif

#ifdef MEG
__global__ void ssspEdgeProcMEG(int super_step)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= GRAPH_STORAGE_GPU::d_vertexArray.size)
		return;
	if(GRAPH_STORAGE_GPU::d_vertexArray.d_updated[tid])
	{	
		int edge_count = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_count[tid];
		//if(tid <1)
		//printf("edge_count = %d\n",edge_count);
		MVT distance = GRAPH_STORAGE_GPU::d_vertexArray.d_distance[tid];

		int fetch = tid;
		for(int i = 0; i < edge_count; i ++)
		{	
			//if(tid <1)
			//	printf("fetch = %d msg_dst = %d\n",fetch,GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[fetch]);
			GRAPH_STORAGE_GPU::d_messageArray.d_val[GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[fetch]] = distance + GRAPH_STORAGE_GPU::d_edgeArray.d_weight[fetch];
			fetch += GRAPH_STORAGE_GPU::d_edgeArray.d_edgeOffset[i];

		}
	}
}
#endif



#ifdef HY
__global__ void ssspEdgeProcHY(int super_step)
{

	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid > GRAPH_STORAGE_GPU::d_vertexArray.size)
		return;
	if(GRAPH_STORAGE_GPU::d_vertexArray.d_updated[tid])
	{
		int edge_count = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_count[tid];
	//	printf("edge_count = %d\n",edge_count);
		MVT distance = GRAPH_STORAGE_GPU::d_vertexArray.d_distance[tid];

		int stride = GRAPH_STORAGE_GPU::d_vertexArray.size;

		int fetch = tid;

		for(int i = 0; i < edge_count; i ++)
		{

			//if(GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[fetch] >= GRAPH_STORAGE_GPU::d_edgeArray.size)
			//	printf("msg dst= %d\n", GRAPH_STORAGE_GPU::d_edgeArray.d_dstVertexID[fetch]);
			GRAPH_STORAGE_GPU::d_messageArray.d_val[GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[fetch]] = distance + GRAPH_STORAGE_GPU::d_edgeArray.d_weight[fetch];
			//e.sendMsg(distance + e.get_weight());
			fetch += stride;
		}

		int start_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid];
		int end_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid + 1];
		//if(end_index - start_index > 100)
		//	printf("tid = %d [%d %d]\n",tid,start_index,end_index);
		for(; start_index < end_index; start_index ++)
		{
			//if(GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[fetch] >= GRAPH_STORAGE_GPU::d_edgeArray.size)
			//	printf("msg dst= %d\n", GRAPH_STORAGE_GPU::d_edgeArray.d_dstVertexID[start_index]);
			GRAPH_STORAGE_GPU::d_messageArray.d_val[GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[start_index]] = distance + GRAPH_STORAGE_GPU::d_edgeArray.d_weight[start_index];

			//e.sendMsg(distance + e.get_weight());
		}	
	}

}
#endif

/**
*	For testing using different data structures
*
* @param   - 
* @return	
* @note	
*
*/
#define TEST_LIST
#ifdef TEST_LIST
void SSSP_EdgeList_Processor()
{
	int edgelist_num = MGLOBAL::gpu_def[0].vertexArray.size;
	//printf("GRAPH_STORAGE_CPU::vertexArray.size = %d\n",GRAPH_STORAGE_CPU::vertexArray.size);
	int blockX = 256;
	int gridX = edgelist_num/blockX;
	if(edgelist_num%blockX)
		gridX ++;
	if(gridX > 65535)
	{
		printf("too many edges, abort\n");
		exit(-1);
	}
	MyCheckErrorMsg("before edgelist");
	//printf("<%d,%d>",gridX, blockX);


#if defined(HY)
	ssspEdgeProcHY<<<gridX, blockX>>>(MGLOBAL::super_step);
#elif defined(MEG)
	ssspEdgeProcMEG<<<gridX, blockX>>>(MGLOBAL::super_step);
#elif defined(AA)
	ssspEdgeProcAA<<<gridX, blockX>>>(MGLOBAL::super_step);
#endif
	cudaThreadSynchronize();
	MyCheckErrorMsg("after edge list");
}
#endif




//---------------------------------------------------------------------------------------------------------//
/**
* update the distance attribute of the vertex
* Vertex Processor
*/
struct UpdateVertex
{
	__device__ void operator() (D_Vertex vertex, int super_step)
	{
		MVT msg_min = vertex.get_combined_msg();
		//printf("%d min dis = %f\n",vertex.index, msg_min);
		if(msg_min < vertex.get_distance())
		{
			vertex.set_distance(msg_min);
			//if(vertex.index == 1)
			//printf("update 1 to %f\n", msg_min);
			vertex.set_updated(true);
			Medusa_Continue();
		}
		else
			vertex.set_updated(false);
	}
};


/**
* Send src's distance plus the lenght of the edge if the src is updated
*/
struct SendMsg
{
	__device__ void operator() (D_Edge e, int super_step)
	{
		int src = e.get_srcVertexID();
		D_Vertex srcV(src);
		float weight = e.get_weight();
		//printf("%f\n", weight);

		if(srcV.get_updated())
		{
			e.sendMsg(srcV.get_distance()+e.get_weight());
			//printf("send from %d to %d with %f + %f\n", src, e.get_dstVertexID(), srcV.get_distance(),e.get_weight());
		}
	}
};



SendMsg sm;
UpdateVertex uv;
Message init_msg;


unsigned int init_timer, sm_timer, com_timer, uv_timer;
#define INDIVIDUALTIMING
void Medusa_Exec()
{
#ifdef INDIVIDUALTIMING
	if(MGLOBAL::super_step == 0)
	{
		cutCreateTimer(&init_timer);
		cutCreateTimer(&sm_timer);
		cutCreateTimer(&com_timer);
		cutCreateTimer(&uv_timer);
	}
#endif

	init_msg.val = FLT_MAX;

#ifdef INDIVIDUALTIMING
	cutStartTimer(init_timer);
#endif
	InitMessageBuffer(init_msg);
#ifdef INDIVIDUALTIMING
	cutStopTimer(init_timer);
#endif	


#ifdef INDIVIDUALTIMING
	cutStartTimer(sm_timer);
#endif
	//FunctorHolder<EDGE>::Run(sm);
	SSSP_EdgeList_Processor();
#ifdef INDIVIDUALTIMING
	cutStopTimer(sm_timer);
#endif	
	//combiner

#ifdef INDIVIDUALTIMING
	cutStartTimer(com_timer);
#endif
	MGLOBAL::com.combineAllDevice();
#ifdef INDIVIDUALTIMING
	cutStopTimer(com_timer);
#endif



#ifdef INDIVIDUALTIMING
	cutStartTimer(uv_timer);
#endif
	FunctorHolder<VERTEX>::Run(uv);
#ifdef INDIVIDUALTIMING
	cutStopTimer(uv_timer);

	if(MGLOBAL::super_step == 864)
	{
		float t;
		t = cutGetTimerValue(init_timer);
		printf("init %f ms\n",t);
		t = cutGetTimerValue(sm_timer);
		printf("sm %f ms\n",t);
		t = cutGetTimerValue(com_timer);
		printf("com %f ms\n",t);
		t = cutGetTimerValue(uv_timer);
		printf("uv %f ms\n",t);
	}
#endif
	//exit(-1);
}


#endif
