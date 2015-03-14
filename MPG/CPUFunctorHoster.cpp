#include "CPUFunctorHoster.h"
#include "omp.h"
#include "Configuration.h"


bool cont[24];
MessageArray messageArray;

/**
*
*
* @param vertex_index - vertex_index-th edge will be processed by this function
* @return	
* @note	
*
*/
void PG_Vertex_Processor_UpdateVertex(int vertex_index, int super_step, VertexArray &va, EdgeArray &ea)
{
	//get combined msg

	MVT sum = 0;
	for(int i = va.msg_index[vertex_index]; i < va.msg_index[vertex_index + 1]; i ++)
	{
//		printf("vertex_index = %d   i = %d\n",vertex_index, i);
		sum += messageArray.val[i];
	}
	//printf("[%d]msg sum:%f\n",vertex_index, sum);
	sum *= 0.85;
	va.level[vertex_index] = 0.15 + sum;

}


/**
*
*
* @param   - 
* @return	
* @note	
*
*/
void PG_Edge_Processor_SendMsg(int edge_index, int super_step, VertexArray &va, EdgeArray &ea)
{
	int src_id = ea.srcVertexID[edge_index];
	MVT level = va.level[src_id];
	int edge_count = va.edge_count[src_id];
	messageArray.val[ea.msgDstID[edge_index]] = level/edge_count;
	//printf("[%d]%f\n",ea.msgDstID[edge_index],level/edge_count);
}


/**
*
*
* @param   - 
* @return	
* @note	
*
*/
void PG_Vertex_Processor_SendMsg(int vertex_index, int super_step, VertexArray &va, EdgeArray &ea)
{

#ifdef AA
	MVT level = va.level[vertex_index];

	int edge_start = va.edge_index[vertex_index];
	int edge_end = va.edge_index[vertex_index + 1];
	int edge_count = edge_end - edge_start;

	for(int edge_index = edge_start; edge_index < edge_end; edge_index ++)
	{
		messageArray.val[ea.msgDstID[edge_index]] = level/edge_count;
		//printf("[%d]%f\n",ea.msgDstID[edge_index],level/edge_count);
	}

#endif


#ifdef MEG
	MVT level = va.level[vertex_index];
	int edge_count = va.edge_count[vertex_index];
	MVT msg = level/edge_count;
	int fetch = vertex_index;
	//printf("edge count:%d\n", edge_count);
	for(int i = 0; i < edge_count; i ++)
	{
		int msg_dst_index = ea.msgDstID[fetch];
		//printf("fetch = %d msg_dst_index = %d\n",fetch, msg_dst_index);
		messageArray.val[msg_dst_index] = msg;		
		fetch += ea.edgeOffset[i];
	}

#endif

#ifdef HY
	int edge_count = va.edge_count[vertex_index];
	int start_index = va.edge_index[vertex_index];
	int end_index = va.edge_index[vertex_index + 1];
	//calculate message
	MVT level = va.level[vertex_index];
	MVT msg = level/(edge_count + end_index - start_index);


	int stride = va.size;
	int fetch = vertex_index;

	for(int i = 0; i < edge_count; i ++)
	{
		//send message
		int msg_dst_index = ea.msgDstID[fetch];
		//printf("msg_dst_index = %d\n",msg_dst_index);
		//	if(msg_dst_index  >= GRAPH_STORAGE_GPU::d_messageArray.size)
		//		printf("msg_dst_index = %d\n",msg_dst_index);
		messageArray.val[msg_dst_index] = msg;
		fetch += stride;
	}


	for(; start_index < end_index; start_index ++)
	{
		//send message
		int msg_dst_index = ea.msgDstID[start_index];
		//	if(msg_dst_index  >= GRAPH_STORAGE_GPU::d_messageArray.size)
		//		printf("msg_dst_index = %d\n",msg_dst_index);
		//	printf("msg_dst_index = %d\n",msg_dst_index);
		messageArray.val[msg_dst_index] = msg;

	}

#endif


}




void Init_CPU_Medusa(EdgeArray &ea)
{
	messageArray.resize(ea.size);
}

void Medusa_Exec_CPU(VertexArray &va, EdgeArray &ea)
{
	int super_step = 0;
	for(int i = 0; i < 100; i ++)
	{

		//init buffer
		#pragma omp parallel for
		for(int j = 0; j < ea.size; j ++)
			messageArray.val[j] = 0;

		#pragma omp parallel for
		for(int j = 0; j < va.size; j ++)
			PG_Vertex_Processor_SendMsg(j, super_step, va, ea);

		#pragma omp parallel for
		for(int j = 0; j < va.size; j ++)
			PG_Vertex_Processor_UpdateVertex(j,super_step, va, ea);

		super_step ++;
	}
	float sum = 0;
	for(int i = 0; i < va.size; i ++)
		sum += va.level[i];
	//printf("CPU sum = %f\n",sum);
	//messageArray.resize(0);
}
