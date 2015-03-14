#ifndef VERTEXDATATYPE_CU
#define VERTEXDATATYPE_CU

#include "VertexDataType.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include "../MedusaRT/Utilities.h"
#include "../MedusaRT/GraphConverter.h"

VertexArray::VertexArray()
{
	size = 0;
	distance = NULL;
	msg_index = NULL;
	edge_count = NULL;
	updated = NULL;
#if defined(HY)
	edge_index = NULL;
#elif defined(AA)
	edge_index = NULL;
#endif
}

void VertexArray::resize(int num)
{
	if(size)
	{
		free(distance);
		free(msg_index);
		free(edge_count);
		free(updated);
#if defined(HY)
		free(edge_index);
#elif defined(AA)
		free(edge_index);
#endif
	}
	size = num;
	CPUMalloc((void**)&distance, sizeof(MVT)*size);
	CPUMalloc((void**)&msg_index, sizeof(int)*(size+1));
	CPUMalloc((void**)&edge_count, sizeof(int)*size);
	CPUMalloc((void**)&updated, sizeof(bool)*size);
#if defined(HY)
	CPUMalloc((void**)&edge_index, sizeof(int)*(size+1));
#elif defined(AA)
	CPUMalloc((void**)&edge_index, sizeof(int)*(size+1));
#endif
}

void VertexArray::assign(int i, Vertex v)
{
	distance[i] = v.distance;
	msg_index[i] = v.msg_index;
	edge_count[i] = v.edge_count;
	updated[i] = v.updated;
	//printf("edge_count[%d] = %d\n",i,edge_count[i]);

#if defined(AA)
	edge_index[i] = v.edge_index;
#elif defined(HY)
	edge_index[i] = v.edge_index;
#endif
}

void VertexArray::build(GraphIR &graph)
{
	//construct vertex array
	resize(graph.vertexNum);
	for(int i = 0; i < graph.vertexNum; i ++)
	{
		assign(i, graph.vertexArray[i].vertex);
	}
	//construct msg_index
	//compute prefix sum
	
	msg_index[0] = 0;
	for(int i = 1; i <= graph.vertexNum; i ++)
		msg_index[i] = graph.vertexArray[i-1].incoming_edge_count + msg_index[i-1];		 
#ifdef AA
	edge_index[0] = 0;
	for(int i = 1; i <= graph.vertexNum; i ++)
	{
		edge_index[i] = graph.vertexArray[i-1].vertex.edge_count + edge_index[i-1];
	}
#endif
}


void D_VertexArray::Fill(VertexArray &varr)
{
	if(size != 0)
	{
		CUDA_SAFE_CALL(cudaFree(d_distance));
		CUDA_SAFE_CALL(cudaFree(d_msg_index));
		CUDA_SAFE_CALL(cudaFree(d_edge_count));
		CUDA_SAFE_CALL(cudaFree(d_updated));
#if  defined(HY)
		CUDA_SAFE_CALL(cudaFree(d_edge_index));
#elif defined(AA)
		CUDA_SAFE_CALL(cudaFree(d_edge_index));
#endif
	}
	size = varr.size;
	GPUMalloc((void**)&d_distance, sizeof(MVT)*size);
	GPUMalloc((void**)&d_msg_index, sizeof(int)*(size+1));
	GPUMalloc((void**)&d_edge_count, sizeof(int)*size);
	GPUMalloc((void**)&d_updated, sizeof(bool)*size);
	CUDA_SAFE_CALL(cudaMemcpy(d_distance, varr.distance, sizeof(MVT)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_msg_index, varr.msg_index, sizeof(int)*(size+1), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_count, varr.edge_count, sizeof(int)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_updated, varr.updated, sizeof(bool)*size, cudaMemcpyHostToDevice));
#if defined(HY)
	GPUMalloc((void**)&d_edge_index, sizeof(int)*(size+1));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_index, varr.edge_index, sizeof(int)*(size+1), cudaMemcpyHostToDevice));
#elif defined(AA)
	GPUMalloc((void**)&d_edge_index, sizeof(int)*(size+1));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_index, varr.edge_index, sizeof(int)*(size+1), cudaMemcpyHostToDevice));
#endif


}


void D_VertexArray::Free()
{
	if(size != 0)
	{
		CUDA_SAFE_CALL(cudaFree(d_distance));
		CUDA_SAFE_CALL(cudaFree(d_msg_index));
		CUDA_SAFE_CALL(cudaFree(d_edge_count));
		CUDA_SAFE_CALL(cudaFree(d_updated));
#if  defined(HY)
		CUDA_SAFE_CALL(cudaFree(d_edge_index));
#elif defined(AA)
		CUDA_SAFE_CALL(cudaFree(d_edge_index));
#endif
	}
}

void D_VertexArray::Dump(VertexArray &varr)
{

	if(size != varr.size)
	{
		if(varr.size)
		{
			free(varr.distance);
			free(varr.msg_index);
			free(varr.edge_count);
			free(varr.updated);
		}
		CPUMalloc((void**)&varr.distance, size*sizeof(MVT));
		CPUMalloc((void**)&varr.msg_index, size*sizeof(int));
		CPUMalloc((void**)&varr.edge_count, size*sizeof(int));
		CPUMalloc((void**)&varr.updated, size*sizeof(bool));
	}
	CUDA_SAFE_CALL(cudaMemcpy(varr.distance, d_distance, sizeof(MVT)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.msg_index, d_msg_index, sizeof(int)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.edge_count, d_edge_count, sizeof(int)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.updated, d_updated, sizeof(bool)*size, cudaMemcpyDeviceToHost));
}

#endif
