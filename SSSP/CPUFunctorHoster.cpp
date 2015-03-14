#include "CPUFunctorHoster.h"
#include <cfloat>
#include <omp.h>

bool cont[24];

MessageArray messageArray;


/**
* update the distance attribute of the vertex
* Vertex Processor
*/

void SSSP_Vertex_Processor_UpdateVertex(int vertex_index, int super_step, VertexArray &va, EdgeArray &ea)
{
	MVT msg_min = FLT_MAX;
	for(int i = va.msg_index[vertex_index]; i < va.msg_index[vertex_index + 1]; i ++)
	{
		if(messageArray.val[i] < msg_min)
			msg_min = messageArray.val[i];
	}
	if(msg_min < va.distance[vertex_index])
	{
		va.distance[vertex_index] = msg_min;
//		va.updated[vertex_index] = true;
		cont[omp_get_thread_num()] = true;
		//printf("update cont = %s\n", cont?"true":"false");
	}
//	else
//		va.updated[vertex_index] = false;

}



/**
* Send src's distance plus the lenght of the edge if the src is updated
*/
void SSSP_Edge_Processor_SendMsg(int edge_index, int super_step, VertexArray &va, EdgeArray &ea)
{
	int src = ea.srcVertexID[edge_index];
	float weight = ea.weight[edge_index];
//	if(va.updated[src])
//	{
		int msg_dst = ea.msgDstID[edge_index];
		messageArray.val[msg_dst] = va.distance[src] + ea.weight[edge_index];
//	}
}


void Init_CPU_Medusa(int size)
{
	messageArray.resize(size);
}


void Medusa_Exec_CPU(VertexArray &va, EdgeArray &ea)
{
	int super_step = 0;
	while(true)
	{
		for(int i = 0; i < 24; i ++)
			cont[i] = false;

		#pragma omp parallel for
		for(int i = 0; i < ea.size; i ++)
			messageArray.val[i] = FLT_MAX;

		#pragma omp parallel for
		for(int i = 0; i < ea.size; i ++)
		{
			SSSP_Edge_Processor_SendMsg(i, super_step, va, ea);
		}

		#pragma omp parallel for
		for(int i = 0; i < va.size; i ++)
		{
			SSSP_Vertex_Processor_UpdateVertex(i, super_step, va, ea);
		}

		super_step ++;

		bool agg_cont = false;
                for(int i = 0; i < 24; i ++)
                        if(cont[i])
                        {
                                agg_cont = true;
                                break;
                        }
                if(!agg_cont)
                        break;


	}
	printf("CPU super step %d\n",super_step);
//	exit(-1);
	
}
