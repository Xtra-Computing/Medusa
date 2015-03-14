#ifndef VERTEXDATATYPE_CU
#define VERTEXDATATYPE_CU

#include "VertexDataType.h"
#include "../MedusaRT/GraphStorage.h"
#include "../MedusaRT/Utilities.h"
#include "../MedusaRT/GraphConverter.h"
#include <cutil_inline_runtime.h>

VertexArray::VertexArray()
{
	size = 0;
	level = NULL;
	msg_index = NULL;
	edge_count = NULL;
	pg_edge_num = NULL;

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
		free(msg_index);
		free(edge_count);
		free(pg_edge_num);
#if defined(HY)
		free(edge_index);
#elif defined(AA)
		free(edge_index);
#endif
	}
	size = num;
	CPUMalloc((void**)&level, sizeof(MVT)*size);
	CPUMalloc((void**)&msg_index, sizeof(int)*(size+1));
	CPUMalloc((void**)&edge_count, sizeof(int)*size);
	CPUMalloc((void**)&pg_edge_num, sizeof(int)*size);


#if defined(HY)
	CPUMalloc((void**)&edge_index, sizeof(int)*(size+1));
#elif defined(AA)
	CPUMalloc((void**)&edge_index, sizeof(int)*(size+1));
#endif

}

void VertexArray::assign(int i, Vertex v)
{
	level[i] = v.level;
	msg_index[i] = v.msg_index;
	edge_count[i] = v.edge_count;
	pg_edge_num[i] = v.pg_edge_num;

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
		CUDA_SAFE_CALL(cudaFree(d_level));
		CUDA_SAFE_CALL(cudaFree(d_msg_index));
		CUDA_SAFE_CALL(cudaFree(d_edge_count));
		CUDA_SAFE_CALL(cudaFree(d_pg_edge_num));

#if  defined(HY)
		CUDA_SAFE_CALL(cudaFree(d_edge_index));
#elif defined(AA)
		CUDA_SAFE_CALL(cudaFree(d_edge_index));
#endif

	}
	size = varr.size;
	GPUMalloc((void**)&d_level, sizeof(MVT)*size);
	GPUMalloc((void**)&d_msg_index, sizeof(int)*(size+1));
	GPUMalloc((void**)&d_edge_count, sizeof(int)*size);
	GPUMalloc((void**)&d_pg_edge_num, sizeof(int)*size);

	
	CUDA_SAFE_CALL(cudaMemcpy(d_level, varr.level, sizeof(MVT)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_msg_index, varr.msg_index, sizeof(int)*(size+1), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_edge_count, varr.edge_count, sizeof(int)*size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(d_pg_edge_num, varr.pg_edge_num, sizeof(int)*size, cudaMemcpyHostToDevice));


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
		varr.resize(size);
	cutilCheckMsg("before vertex array dump");
	CUDA_SAFE_CALL(cudaMemcpy(varr.level, d_level, sizeof(MVT)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.msg_index, d_msg_index, sizeof(int)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.edge_count, d_edge_count, sizeof(int)*size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(varr.pg_edge_num, d_pg_edge_num, sizeof(int)*size, cudaMemcpyDeviceToHost));

}

#endif