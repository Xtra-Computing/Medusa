
#ifndef USEROPERATORS_H
#define USEROPERATORS_H
#include "../MedusaRT/SystemLibGPU.h"
#include "Configuration.h"



//---------------------------------------------------------------------------------------------------------//
/**
*
* @param []	
* @return	
* @note	
*
*/
struct UpdateLevel
{
	__device__ void operator() (D_Edge e, int super_step)
	{
		int src_id = e.get_srcVertexID();
		//if(src_id == 5469)
		//	printf("Got root\n");
		D_Vertex src_V(src_id);
		if(src_V.get_level() == super_step)
		{
			D_Vertex dst_V(e.get_dstVertexID());
			if(dst_V.get_level() == MVT_Init_Value)
			{
				dst_V.set_level(super_step + 1);
				Medusa_Continue();
			}
		}
	}
};
UpdateLevel ul;







void Medusa_Exec()
{

	//cudaFuncSetCacheConfig(bfsEdgeProc, cudaFuncCachePreferL1);
	if(MGLOBAL::num_gpu_to_use > 1 && MGLOBAL::max_hop >= 1)
		MGFunctorHolder<MGMH_EDGE>::Run(ul);
	else if(MGLOBAL::num_gpu_to_use > 1)
		MGFunctorHolder<MG_EDGE>::Run(ul);
	else
		FunctorHolder<EDGE>::Run(ul);
	//BFS_Edge_Processor();
	//BFS_EdgeList_Processor();
	//
	//
}


#endif
