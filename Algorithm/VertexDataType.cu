#ifndef VERTEXDATATYPE_CU
#define VERTEXDATATYPE_CU

#include "VertexDataType.h"
#include "../MedusaRT/GraphStorage.h"
#include "../MedusaRT/Utilities.h"
#include "../MedusaRT/GraphConverter.h"

VertexArray::VertexArray()
{
	size = 0;
	level = NULL;
	edge_count = NULL;

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
		free(level);
		free(edge_count);

#if defined(HY)
		free(edge_index);
#elif defined(AA)
		free(edge_index);
#endif
	}
	size = num;
	CPUMalloc((void**)&level, sizeof(MVT)*size);
	CPUMalloc((void**)&edge_count, sizeof(int)*size);

#if defined(HY)
	CPUMalloc((void**)&edge_index, sizeof(int)*(size+1));
#elif defined(AA)
	CPUMalloc((void**)&edge_index, sizeof(int)*(size+1));
#endif
}

void VertexArray::assign(int i, Vertex v)
{
	level[i] = v.level;
	edge_count[i] = v.edge_count;

#if defined(AA)
	edge_index[i] = v.edge_index;
#elif defined(HY)
	edge_index[i] = v.edge_index;
#endif
	//printf("edge_count[%d] = %d\n",i,edge_count[i]);

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
		CUDA_SAFE_CALL(cudaFree(d_level));
		CUDA_SAFE_CALL(cudaFree(d_edge_count));

#if  defined(HY)
		CUDA_SAFE_CALL(cudaFree(d_edge_index));
#elif defined(AA)
		CUDA_SAFE_CALL(cudaFree(d_edge_index));
#endif
	}
	size = varr.size;
	GPUMalloc((void**)&d_level, sizeof(MVT)*size);
	GPUMalloc((void**)&d_edge_count, sizeof(int)*size);
	CUDA_SAFE_CALL(cudaMemcpy(d_level, varr.level, sizeof(MVT)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_count, varr.edge_count, sizeof(int)*size, cudaMemcpyHostToDevice));


#if defined(HY)
	GPUMalloc((void**)&d_edge_index, sizeof(int)*(size+1));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_index, varr.edge_index, sizeof(int)*(size+1), cudaMemcpyHostToDevice));
#elif defined(AA)
	GPUMalloc((void**)&d_edge_index, sizeof(int)*(size+1));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_index, varr.edge_index, sizeof(int)*(size+1), cudaMemcpyHostToDevice));
#endif



}

void D_VertexArray::Dump(VertexArray &varr)
{

	if(size != varr.size)
	{
		if(varr.size)
		{
			free(varr.level);
			free(varr.edge_count);
		}
		varr.size = size;
		CPUMalloc((void**)&varr.level, size*sizeof(MVT));
		CPUMalloc((void**)&varr.edge_count, size*sizeof(int));
	}
	CUDA_SAFE_CALL(cudaMemcpy(varr.level, d_level, sizeof(MVT)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.edge_count, d_edge_count, sizeof(int)*size, cudaMemcpyDeviceToHost));
}

#endif