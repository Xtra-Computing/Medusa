#include "CPUFunctorHoster.h"
#include <omp.h>
bool cont[24];

/**
*
*
* @param edge_index - edge_index-th edge will be processed by this function
* @return	
* @note	
*
*/
int total_edges_traversed = 0;
int vertex_frontier_size = 0;
int edge_frontier_size = 0;
void BFS_Edge_Processor(int edge_index, int super_step, VertexArray &va, EdgeArray &ea)
{
	int src_id = ea.srcVertexID[edge_index];
	if(va.level[src_id] == super_step)
	{
		total_edges_traversed ++;
		int dst_id = ea.dstVertexID[edge_index];
		edge_frontier_size ++;
		if(va.level[dst_id] == MVT_Init_Value)
		{
			va.level[dst_id] = super_step + 1;
	    	cont[omp_get_thread_num()] = true;
			vertex_frontier_size ++;
		}
	}
}



void Medusa_Exec_CPU(VertexArray &va, EdgeArray &ea)
{
	int super_step = 0;
	while(true)
	{
		vertex_frontier_size = 0;
		edge_frontier_size = 0;
		for(int i = 0; i < 24; i ++)
			cont[i] = false;
		//#pragma omp parallel for
		for(int i = 0; i < ea.size; i ++)
		{
			BFS_Edge_Processor(i, super_step, va, ea);
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
	printf("CPU total traversed edge %d with %d steps\n", total_edges_traversed, super_step);
}
