#ifndef MULTIAPIHOSTERS_H
#define MULTIAPIHOSTERS_H

#include "../MedusaRT/APIKernelWrappers.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include <cutil_inline.h>
#include "../Algorithm/Configuration.h"
#include "../MultipleGPU/PartitionManager.h"
#include "../MultipleGPU/Gather.cu"
#include "../MultipleGPU/WEQueue.h"



enum EMVAPIType {EDGE, EDGELIST, MESSAGE, MESSAGELIST, VERTEX, MG_VERTEX, EDGE_HT, MG_EDGE, MGMH_EDGE, MGMH_VERTEX};


/* Queue related API */
enum QAPIType{EQ2EQ, EQ2VQ, EQConsume, /* work on edge queue  */
			VQ2EQ, VQ2VQ, VQConsume, /* work on vertex queue */
			VA2EQ, VA2AQ, /* work on vertex array  */
			EA2EQ, EA2AQ  /* work on edge array */
};

template<EMVAPIType f>
struct MGFunctorHolder{};


/**
* Synchronized edge operator, no need read updated vertex data
*
* @param   - 
* @return	
* @note	
*
*/


#define UPDATE_REPLICA
//#define MedusaAsyncCopy(dst_addr, source_addr, size, direction, stream) cudaMemcpy(dst_addr, source_addr, size, direction)
#define MedusaAsyncCopy(dst_addr, source_addr, size, direction, stream) cudaMemcpyAsync(dst_addr, source_addr, size, direction, stream)

//define to enable masuring time cost of each component
//#define MEASURE_TIME


/**
*	Asynchronous version of the EDGE holder. The function is returned immediately after
*	operations on (original + core) edges is done. Data exchange start at the same time
*	as the edge operation (in a different thread). EDGE operator on replica's edges is 
*	executed when EdgeOperationSync() is called.
* @param   - 
* @return	
* @note	
*
*/


#ifdef MULTIGPU
template<>
struct MGFunctorHolder<MG_EDGE>
{

	template<class OP>
	static void Run(OP op)
	{
		#ifdef MEASURE_TIME
		unsigned int timer;
		float elapsed_time;

		cutCreateTimer(&timer);
		cutStartTimer(timer);
		#endif

#ifdef UPDATE_REPLICA
		//gather boundary vertices before everything starts (later can consider kernel overlapping)
		for(int gpu_id = 0; gpu_id < MGLOBAL::num_gpu_to_use; gpu_id ++)
		{
			medusaSetDevice(gpu_id);
			int blockX = MGLOBAL::gpu_def[gpu_id].device_prop.multiProcessorCount * 6;
			gather_kernel<<<blockX, 256>>>(MGLOBAL::gpu_def[gpu_id].d_gather_buffer, MGLOBAL::gpu_def[gpu_id].d_vertexArray.d_level, MGLOBAL::gpu_def[gpu_id].d_gather_table, blockX*256, MGLOBAL::gpu_def[gpu_id].gather_table_size);

		}
#endif

		for(int gpu_id = 0; gpu_id < MGLOBAL::num_gpu_to_use; gpu_id ++)
		{

			medusaSetDevice(gpu_id);
			cudaDeviceSynchronize();
			DBGPrintf("GPU %d gather table size = %d\n", gpu_id, MGLOBAL::gpu_def[gpu_id].gather_table_size);
		}

		#ifdef MEASURE_TIME
		cutStopTimer(timer);
		elapsed_time = cutGetTimerValue(timer);
		DBGPrintf("Gather time:%f ms\n", elapsed_time);
		cutResetTimer(timer);
		cutStartTimer(timer);
		#endif


		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);

			int edge_num = MGLOBAL::gpu_def[i].replica_edge_index[0];

			//printf("non-replica edge_num = %d\n",edge_num);
	

			int gridX = MGLOBAL::gpu_def[i].device_prop.multiProcessorCount*6;


			//cudaFuncSetCacheConfig("edgeProcWrapper", cudaFuncCachePreferL1);
			OOEdgeProcWrapper<<<gridX, 256, 0, MGLOBAL::gpu_def[i].sync_stream>>>(MGLOBAL::super_step, op, edge_num, gridX*256);
			MyCheckErrorMsg("after OOEdgeProcWrapper");


#ifdef UPDATE_REPLICA
			//GPU -> CPU
			int original_buffer_offset = 0;
			for(int o_i = 0; o_i < i; o_i ++)
				original_buffer_offset += MGLOBAL::gpu_def[o_i].gather_table_size;
			MedusaAsyncCopy(&MGLOBAL::original_buffer[original_buffer_offset], MGLOBAL::gpu_def[i].d_gather_buffer,sizeof(MVT)*MGLOBAL::gpu_def[i].gather_table_size, cudaMemcpyDeviceToHost, MGLOBAL::gpu_def[i].async_stream);
#endif
			MyCheckErrorMsg("after MedusaAsyncCopy");
		}





//------------------------------------------------------------------------


#ifdef UPDATE_REPLICA	
		//should be multi-threaded since cudaStreamSynchronize take different amount of time
		//for different GPUs
		for(int src_gpu_id = 0; src_gpu_id < MGLOBAL::num_gpu_to_use; src_gpu_id ++)
		{
			medusaSetDevice(src_gpu_id);

			MyCheckErrorMsg("Before sync copy to");
			cudaStreamSynchronize(MGLOBAL::gpu_def[src_gpu_id].async_stream);
			MyCheckErrorMsg("After cudaStreamSynchronize");

			//copy to other GPUs one by one
			int base_original_buffer_offset = 0;
			int original_buffer_offset = 0;
			for(int ci = 0; ci < src_gpu_id; ci ++)
			{
				base_original_buffer_offset += MGLOBAL::gpu_def[ci].gather_table_size;
			}
			for(int dst_gpu_id = 0; dst_gpu_id < MGLOBAL::num_gpu_to_use; dst_gpu_id ++)
			{
				if(src_gpu_id == dst_gpu_id)
					continue; //no self-to-self communication
				medusaSetDevice(dst_gpu_id);
				
				original_buffer_offset = base_original_buffer_offset + MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id];
				//printf("original_buffer_offset = %d\n",original_buffer_offset);
				int scatter_buffer_offset = MGLOBAL::gpu_def[dst_gpu_id].scatter_seg_index[src_gpu_id];
				//printf("scatter_buffer_offset = %d\n",scatter_buffer_offset);
				int cpy_ele_num = (MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id + 1] - MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id]);
				//printf("cpy_ele_num = %d - %d\n",MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id + 1], MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id]);
				if(!cpy_ele_num)
					continue;
				MyCheckErrorMsg("before async copy to");
				DBGPrintf("GPU %d buffer offset %d, origin offset %d, element num %d\n",dst_gpu_id, scatter_buffer_offset, original_buffer_offset, cpy_ele_num);
				DBGPrintf("scatter_buffer_size = %d, original buffer size = %d\n", MGLOBAL::gpu_def[dst_gpu_id].scatter_table_size, MGLOBAL::original_buffer_size);
				MedusaAsyncCopy( &MGLOBAL::gpu_def[dst_gpu_id].d_scatter_buffer[scatter_buffer_offset], &MGLOBAL::original_buffer[original_buffer_offset], sizeof(MVT)*cpy_ele_num, cudaMemcpyHostToDevice, MGLOBAL::gpu_def[dst_gpu_id].async_stream);
				
				MyCheckErrorMsg("After async copy to");
			}
		}
#endif
//----------------------------------------------------------------------------------------

		#ifdef MEASURE_TIME
		//for timing purpose, can be commented		
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			cudaDeviceSynchronize();
		}
		//end of timing purpose

		cutStopTimer(timer);
		elapsed_time = cutGetTimerValue(timer);
		DBGPrintf("Non-replica edge processing and data exchange:%f ms\n",elapsed_time);
		MyCheckErrorMsg("After copy back");



		//apply edge processor to replicas
		cutResetTimer(timer);
		cutStartTimer(timer);
		#endif


#ifdef UPDATE_REPLICA				
		//scatter the results when copy-to-gpu and edge operator on original is done


		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			int block_num = MGLOBAL::gpu_def[i].device_prop.multiProcessorCount * 6;
			scatter_kernel<<<block_num, 256>>>(MGLOBAL::gpu_def[i].d_vertexArray.d_level, MGLOBAL::gpu_def[i].d_scatter_buffer, MGLOBAL::gpu_def[i].d_scatter_table, block_num*256, MGLOBAL::gpu_def[i].scatter_table_size);
			MyCheckErrorMsg("After gather_kernel");
		}


#endif


		#ifdef MEASURE_TIME
		//for timing purpose, can be commented		
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			cudaDeviceSynchronize();
			DBGPrintf("GPU %d scatter table size %d\n", i, MGLOBAL::gpu_def[i].scatter_table_size);
		}
		//end of timing purpose

		cutStopTimer(timer);
		elapsed_time = cutGetTimerValue(timer);
		DBGPrintf("Scatter:%f ms\n",elapsed_time);
		MyCheckErrorMsg("After copy back");



		//apply edge processor to replicas
		cutResetTimer(timer);
		cutStartTimer(timer);



		cutResetTimer(timer);
		cutStartTimer(timer);
		#endif



		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);

			int edge_num = MGLOBAL::gpu_def[i].replica_edge_index[MGLOBAL::max_hop] - MGLOBAL::gpu_def[i].replica_edge_index[0];
			if(!edge_num)
				continue;
			DBGPrintf("replica edge num = %d\n",edge_num);
			//!!!
			int gridX = MGLOBAL::gpu_def[i].device_prop.multiProcessorCount * 6;
			
			//cudaFuncSetCacheConfig("edgeProcWrapper", cudaFuncCachePreferL1);
			MyCheckErrorMsg("before edgeProcWrapper");
			//printf("<%d,%d>\n",gridX,blockX);
			REdgeProcWrapper<<<gridX, 256>>>(MGLOBAL::super_step, op, MGLOBAL::gpu_def[i].replica_edge_index[0], gridX*256);
			//edgeProcWrapper<<<gridX, blockX>>>(MGLOBAL::super_step, op);
		}
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			//MyCheckErrorMsg("before edgeProcWrapper");
			cudaDeviceSynchronize();
			//MyCheckErrorMsg("edgeProcWrapper");
		}

		#ifdef MEASURE_TIME
		cutStopTimer(timer);
		elapsed_time = cutGetTimerValue(timer);
		DBGPrintf("Edge Processor on replica %f ms\n",elapsed_time);
		cutilCheckMsg("End of edge async");
		#endif

	}

};





/*
	Multiple GPU + Multiple Hop Edge processor
	
	The key idea to multiple hops (H hops) implementation is
	(1) Update replica for every H iterations
	(2) Iteratively reduce the number of edges
*/
#define OLD_SHUFFLE
template<>
struct MGFunctorHolder<MGMH_EDGE>
{

	template<class OP>
	static void Run(OP op)
	{
	
#ifdef MEASURE_TIME		
		unsigned int timer;
		float elapsed_time;

		cutCreateTimer(&timer);
		cutStartTimer(timer);
#endif

		int stage = MGLOBAL::super_step % MGLOBAL::max_hop;

#ifdef UPDATE_REPLICA

		if((stage == 0) && (MGLOBAL::super_step !=0) )
		{
			//if(MGLOBAL::super_step == 0)
			//	break;

			//gather boundary vertices before everything starts (later can consider kernel overlapping)
			for(int gpu_id = 0; gpu_id < MGLOBAL::num_gpu_to_use; gpu_id ++)
			{
				medusaSetDevice(gpu_id);
				int blockX = MGLOBAL::gpu_def[gpu_id].device_prop.multiProcessorCount * 6;
				gather_kernel<<<blockX, 256>>>(MGLOBAL::gpu_def[gpu_id].d_gather_buffer, MGLOBAL::gpu_def[gpu_id].d_vertexArray.d_level, MGLOBAL::gpu_def[gpu_id].d_gather_table, blockX*256, MGLOBAL::gpu_def[gpu_id].gather_table_size);
			}

			for(int gpu_id = 0; gpu_id < MGLOBAL::num_gpu_to_use; gpu_id ++)
			{
				medusaSetDevice(gpu_id);
				cudaDeviceSynchronize();
				DBGPrintf("GPU %d gather table size = %d\n", gpu_id, MGLOBAL::gpu_def[gpu_id].gather_table_size);
			}

#ifdef MEASURE_TIME
			cutStopTimer(timer);
			elapsed_time = cutGetTimerValue(timer);
			printf("Gather time:%f ms\n", elapsed_time);


			cutResetTimer(timer);
			cutStartTimer(timer);
#endif

			for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
			{
				medusaSetDevice(i);
				int original_buffer_offset = 0;
				for(int o_i = 0; o_i < i; o_i ++)
					original_buffer_offset += MGLOBAL::gpu_def[o_i].gather_table_size;
				MedusaAsyncCopy(&MGLOBAL::original_buffer[original_buffer_offset], MGLOBAL::gpu_def[i].d_gather_buffer,sizeof(MVT)*MGLOBAL::gpu_def[i].gather_table_size, cudaMemcpyDeviceToHost, MGLOBAL::gpu_def[i].async_stream);

				MyCheckErrorMsg("after MedusaAsyncCopy");
			}

#ifdef	OLD_SHUFFLE
			//should be multi-threaded since cudaStreamSynchronize take different amount of time
			//for different GPUs
			for(int src_gpu_id = 0; src_gpu_id < MGLOBAL::num_gpu_to_use; src_gpu_id ++)
			{
				medusaSetDevice(src_gpu_id);

				MyCheckErrorMsg("Before sync copy to");
				cudaStreamSynchronize(MGLOBAL::gpu_def[src_gpu_id].async_stream);
				MyCheckErrorMsg("After cudaStreamSynchronize");

				//copy to other GPUs one by one
				int base_original_buffer_offset = 0;
				int original_buffer_offset = 0;
				for(int ci = 0; ci < src_gpu_id; ci ++)
				{
					base_original_buffer_offset += MGLOBAL::gpu_def[ci].gather_table_size;
				}
				for(int dst_gpu_id = 0; dst_gpu_id < MGLOBAL::num_gpu_to_use; dst_gpu_id ++)
				{
					if(src_gpu_id == dst_gpu_id)
						continue; //no self-to-self commnunication
					medusaSetDevice(dst_gpu_id);
					original_buffer_offset = base_original_buffer_offset + MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id];
					//printf("original_buffer_offset = %d\n",original_buffer_offset);
					int scatter_buffer_offset = MGLOBAL::gpu_def[dst_gpu_id].scatter_seg_index[src_gpu_id];
					//printf("scatter_buffer_offset = %d\n",scatter_buffer_offset);
					int cpy_ele_num = (MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id + 1] - MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id]);
					//printf("cpy_ele_num = %d - %d\n",MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id + 1], MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id]);
					if(!cpy_ele_num)
						continue;
					MyCheckErrorMsg("before async copy to");
					//printf("gpu %d buffer offset %d, origin offset %d, element num %d\n",dst_gpu_id, scatter_buffer_offset, original_buffer_offset, cpy_ele_num);
					MedusaAsyncCopy( &MGLOBAL::gpu_def[dst_gpu_id].d_scatter_buffer[scatter_buffer_offset], &MGLOBAL::original_buffer[original_buffer_offset], sizeof(MVT)*cpy_ele_num, cudaMemcpyHostToDevice, MGLOBAL::gpu_def[dst_gpu_id].async_stream);
					MyCheckErrorMsg("After async copy to");
				}
			}
#endif


#ifdef NEW_SHUFFLE
			//should be multi-threaded since cudaStreamSynchronize take different amount of time
			//for different GPUs
			bool *stream_completed = new bool[MGLOBAL::num_gpu_to_use];
			for(int src_gpu_id = 0; src_gpu_id < MGLOBAL::num_gpu_to_use; src_gpu_id ++)
			{
				medusaSetDevice(src_gpu_id);

				MyCheckErrorMsg("Before sync copy to");
				cudaStreamSynchronize(MGLOBAL::gpu_def[src_gpu_id].async_stream);
				MyCheckErrorMsg("After cudaStreamSynchronize");

				//copy to other GPUs one by one
				int original_buffer_offset = 0;
				for(int ci = 0; ci < src_gpu_id; ci ++)
				{
					original_buffer_offset += MGLOBAL::gpu_def[ci].gather_table_size;
				}
				for(int dst_gpu_id = 0; dst_gpu_id < MGLOBAL::num_gpu_to_use; dst_gpu_id ++)
				{
					medusaSetDevice(dst_gpu_id);
					original_buffer_offset += MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id];
					//printf("original_buffer_offset = %d\n",original_buffer_offset);
					int scatter_buffer_offset = MGLOBAL::gpu_def[dst_gpu_id].scatter_seg_index[src_gpu_id];
					//printf("scatter_buffer_offset = %d\n",scatter_buffer_offset);
					int cpy_ele_num = (MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id + 1] - MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id]);
					//printf("cpy_ele_num = %d - %d\n",MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id + 1], MGLOBAL::gpu_def[src_gpu_id].gather_seg_index[dst_gpu_id]);
					if(!cpy_ele_num)
						continue;
					MyCheckErrorMsg("before async copy to");
					//printf("gpu %d buffer offset %d, origin offset %d, element num %d\n",dst_gpu_id, scatter_buffer_offset, original_buffer_offset, cpy_ele_num);
					MedusaAsyncCopy( &MGLOBAL::gpu_def[dst_gpu_id].d_scatter_buffer[scatter_buffer_offset], &MGLOBAL::original_buffer[original_buffer_offset], sizeof(MVT)*cpy_ele_num, cudaMemcpyHostToDevice, MGLOBAL::gpu_def[dst_gpu_id].async_stream);
					MyCheckErrorMsg("After async copy to");
				}
			}
#endif




#ifdef MEASURE_TIME
			cutStopTimer(timer);
			elapsed_time = cutGetTimerValue(timer);
			printf("Data exchange:%f ms\n",elapsed_time);
			MyCheckErrorMsg("After copy back");



			//scatter the results when copy-to-gpu and edge operator on original is done
			cutResetTimer(timer);
			cutStartTimer(timer);
#endif

			for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
			{
				medusaSetDevice(i);
				int block_num = MGLOBAL::gpu_def[i].device_prop.multiProcessorCount * 6;
				DBGPrintf("GPU %d begin to scatter\n", i);
				//int dbg_temp;
				//scanf("%d", &dbg_temp);
				MyCheckErrorMsg("Before scatter_kernel");	
				scatter_kernel<<<block_num, 256>>>(MGLOBAL::gpu_def[i].d_vertexArray.d_level, MGLOBAL::gpu_def[i].d_scatter_buffer, MGLOBAL::gpu_def[i].d_scatter_table, block_num*256, MGLOBAL::gpu_def[i].scatter_table_size);
				cutilCheckMsg("After scatter_kernel");
			}





			for(int gpu_id = 0; gpu_id < MGLOBAL::num_gpu_to_use; gpu_id ++)
			{

				medusaSetDevice(gpu_id);
				cudaDeviceSynchronize();
				DBGPrintf("GPU %d scatter table size = %d\n", gpu_id, MGLOBAL::gpu_def[gpu_id].scatter_table_size);
			}

#ifdef MEASURE_TIME
			cutStopTimer(timer);
			elapsed_time = cutGetTimerValue(timer);
			printf("Scatter time:%f ms\n", elapsed_time);
#endif

		}
#endif


#ifdef MEASURE_TIME
		cutResetTimer(timer);
		cutStartTimer(timer);
#endif

		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			int edge_num = MGLOBAL::gpu_def[i].replica_edge_index[MGLOBAL::max_hop - stage];
			int gridX = MGLOBAL::gpu_def[i].device_prop.multiProcessorCount*6;
			OOEdgeProcWrapper<<<gridX, 256, 0, MGLOBAL::gpu_def[i].sync_stream>>>(MGLOBAL::super_step, op, edge_num, gridX*256);
			MyCheckErrorMsg("after OOEdgeProcWrapper");

		}


		//----------------------------------------------------------------------------------------

		//for timing purpose, can be commented		
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			cudaDeviceSynchronize();
		}
		//end of timing purpose

#ifdef MEASURE_TIME
		cutStopTimer(timer);
		elapsed_time = cutGetTimerValue(timer);
		printf("Edge processing:%f ms\n",elapsed_time);
		MyCheckErrorMsg("After copy back");
#endif

	}

};

/*
	Multiple-hop multiple-GPU vertex processor,
	the only difference from MG_VERTEX is you need
	to iterative decrease the number of vertices
	to be processed according to the phase.


*/
template<>
struct MGFunctorHolder<MGMH_VERTEX>
{
	template<class OP>
	static void Run(OP op)
	{
		int stage = MGLOBAL::super_step % MGLOBAL::max_hop;
		unsigned int timer;
		cutCreateTimer(&timer);
		cutStartTimer(timer);
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			//Vertex Operator only apply to Original and Replica
			//MGMH_VERTEX process one hop less than MGMH_EDGE
			int vertex_num = MGLOBAL::gpu_def[i].replica_index[MGLOBAL::max_hop - stage - 1];
			int gridX = MGLOBAL::gpu_def[i].device_prop.multiProcessorCount*6;
			//cudaFuncSetCacheConfig("vertexProcWrapper", cudaFuncCachePreferL1);
			MyCheckErrorMsg("before vertexProcWrapper");
			OOvertexProcWrapper<<<gridX, 256>>>(MGLOBAL::super_step, op, vertex_num, gridX*256);
		}

		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{	
			medusaSetDevice(i);
			cudaDeviceSynchronize();
			MyCheckErrorMsg("vertexProcWrapper");
		}
		cutStopTimer(timer);
		float elapsed_time = cutGetTimerValue(timer);
		DBGPrintf("MGFunctorHolder<VERTEX>:%f ms\n", elapsed_time);
	}
};

#endif //MULTIGPU


/**
* 
*
* @param   - 
* @return	
* @note	
*
*/

template<>
struct MGFunctorHolder<MESSAGE>
{
	template<class OP>
	static void Run(OP op)
	{
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{	
			int message_num = MGLOBAL::gpu_def[i].edgeArray.size;
			int blockX = 256;
			int gridX = message_num/blockX;
			if(message_num%blockX)
				gridX ++;
			if(gridX > 65535)
			{
				printf("too many edges, abort\n");
				exit(-1);
			}
			messageProcWrapper<<<gridX, blockX,0,MGLOBAL::gpu_def[i].sync_stream>>>(MGLOBAL::super_step, op);
			MyCheckErrorMsg("messageProcWrapper");
		}

		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{	
			medusaSetDevice(i);
			cudaDeviceSynchronize();
			MyCheckErrorMsg("messageProcWrapper");
		}
	}
};



#ifdef MULTIGPU
template<>
struct MGFunctorHolder<MG_VERTEX>
{
	template<class OP>
	static void Run(OP op)
	{
		unsigned int timer;
		cutCreateTimer(&timer);
		cutStartTimer(timer);
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			//Vertex Operator only apply to Original and Replica
			int vertex_num = MGLOBAL::gpu_def[i].replica_index[0];
			DBGPrintf("vertex_num = %d\n", vertex_num);
			//!!!!!
			int gridX = MGLOBAL::gpu_def[i].device_prop.multiProcessorCount*6;
			//cudaFuncSetCacheConfig("vertexProcWrapper", cudaFuncCachePreferL1);
			MyCheckErrorMsg("before vertexProcWrapper");
			OOvertexProcWrapper<<<gridX, 256>>>(MGLOBAL::super_step, op, vertex_num, gridX*256);
		}

		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{	
			medusaSetDevice(i);
			cudaDeviceSynchronize();
			MyCheckErrorMsg("vertexProcWrapper");
		}
		cutStopTimer(timer);
		float elapsed_time = cutGetTimerValue(timer);
		DBGPrintf("MGFunctorHolder<VERTEX>:%f ms\n", elapsed_time);
	}
};

#endif //MULTIGPU



template<EMVAPIType f>
struct FunctorHolder{};


template<>
struct FunctorHolder<EDGE>
{

	template<class OP>
	static void Run(OP op)
	{
		medusaSetDevice(0);
		int edge_num = MGLOBAL::gpu_def[0].edgeArray.size;
		DBGPrintf("MGLOBAL::gpu_def[0].edgeArray.size = %d\n",MGLOBAL::gpu_def[0].edgeArray.size);

		//!!!!
		int gridX = MGLOBAL::gpu_def[0].device_prop.multiProcessorCount * 6;
		//cudaFuncSetCacheConfig("edgeProcWrapper", cudaFuncCachePreferL1);
		//unsigned int timer;
		//cutCreateTimer(&timer);
		//cutStartTimer(timer);
		MyCheckErrorMsg("before edgeProcWrapper");

		edgeProcWrapper<<<gridX, 256>>>(MGLOBAL::super_step, op, gridX*256);
		cudaDeviceSynchronize();
		//cutStopTimer(timer);
		//float elapsed_time = cutGetTimerValue(timer);
		//printf("Edge Processor:%f ms\n",elapsed_time);
		MyCheckErrorMsg("edgeProcWrapper");
	}
};


/* EDGE_HT -- specially used by hash table */
template<>
struct FunctorHolder<EDGE_HT>
{

	template<class OP>
	static void Run(OP op)
	{
		int edge_num = MGLOBAL::gpu_def[0].edgeArray.size;
		//printf("GRAPH_STORAGE_CPU::edgeArray.size = %d\n",GRAPH_STORAGE_CPU::edgeArray.size);
		int blockX = 256;
		int gridX = 60;
		int total_thread_num = blockX * gridX;
		//cudaFuncSetCacheConfig("edgeProcWrapper", cudaFuncCachePreferL1);
		MyCheckErrorMsg("before edgeProcWrapper");
		//printf("<%d,%d>\n",gridX,blockX);
		edgeProcWrapperHT<<<gridX, blockX>>>(MGLOBAL::super_step, op, total_thread_num);
		cudaThreadSynchronize();
		MyCheckErrorMsg("edgeProcWrapper");

	}
};


template<>
struct FunctorHolder<EDGELIST>
{
	template<class OP>
	static void Run(OP op)
	{
		int vertex_num = MGLOBAL::gpu_def[0].vertexArray.size;
		int gridX = MGLOBAL::gpu_def[0].device_prop.multiProcessorCount*6;
		edgeListProcWrapper<<<gridX, 256>>>(MGLOBAL::super_step, op, gridX*256);
		cudaThreadSynchronize();
		MyCheckErrorMsg("edgeListProcWrapper");
	}
};

template<>
struct FunctorHolder<MESSAGE>
{
	template<class OP>
	static void Run(OP op)
	{
		int message_num = MGLOBAL::gpu_def[0].edgeArray.size;
		int blockX = 256;
		int gridX = message_num/blockX;
		if(message_num%blockX)
			gridX ++;
		messageProcWrapper<<<gridX, blockX>>>(MGLOBAL::super_step, op);
		cudaThreadSynchronize();
		MyCheckErrorMsg("messageProcWrapper");
	}
};


template<>
struct FunctorHolder<VERTEX>
{
	template<class OP>
	static void Run(OP op)
	{
		medusaSetDevice(0);
		int vertex_num = MGLOBAL::gpu_def[0].vertexArray.size;
		DBGPrintf("vertex_num = %d\n", vertex_num);
		int gridX = MGLOBAL::gpu_def[0].device_prop.multiProcessorCount*6;
	
		MyCheckErrorMsg("before vertexProcWrapper");
		vertexProcWrapper<<<gridX, 256>>>(MGLOBAL::super_step, op, gridX*256);
		cudaThreadSynchronize();
		MyCheckErrorMsg("vertexProcWrapper");
	}
};



#endif
