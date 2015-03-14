#ifndef PUBLICAPI_H
#define PUBLICAPI_H

#include "../MedusaRT/Utilities.h"
#include "../MultipleGPU/MultiAPIHoster.h"
#include "../Algorithm/EdgeDataType.h"
#include "../Algorithm/VertexDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../Algorithm/Configuration.h"
#include "../MultipleGPU/MultiUtilities.h"
#ifdef VIS
#include "../MedusaRT/CUDAOpenglInterop.h"
#endif


void InitConfig(char *cmdarg, char *cmdarg2, MessageMode mm, bool exe)
{
	MGLOBAL::num_gpu_to_use = atoi(cmdarg);
	MGLOBAL::max_hop = atoi(cmdarg2);
	MGLOBAL::message_mode = mm;

#if defined MEG
	MGLOBAL::graph_ds_type = DS_MEG;
#elif defined AA
	MGLOBAL::graph_ds_type = DS_AA;
#elif defined HY
	MGLOBAL::graph_ds_type = DS_HY;
#elif defined ELL
	MGLOBAL::graph_ds_type = DS_ELL;
#else
	#error Medusa Data Type Must Be Defined (in file MedusaRT/Configuration.h)
#endif


	MGLOBAL::toExecute = exe;
	InitMultiGPU(MGLOBAL::num_gpu_to_use);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
}



void InitHostDS(GraphPartition* gp_array)
{
	//set up data structures for all GPUs
	MGLOBAL::total_edge_count = (int*)malloc(sizeof(int)*MGLOBAL::num_gpu_to_use);
	memset(MGLOBAL::total_edge_count, 0, sizeof(int)*MGLOBAL::num_gpu_to_use);

	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		printf("GPU %d enter\n",i);
		if(MGLOBAL::graph_ds_type == DS_MEG || MGLOBAL::graph_ds_type == DS_HY)
			gp_array[i].graph.sortByDegree();

		MGLOBAL::gpu_def[i].vertexArray.size = 0;
		MGLOBAL::gpu_def[i].edgeArray.size = 0;
		MGLOBAL::gpu_def[i].vertexArray.build(gp_array[i].graph);
		MGLOBAL::gpu_def[i].replica_index = gp_array[i].replica_index;
		MGLOBAL::gpu_def[i].replica_edge_index = gp_array[i].replica_edge_index;


		MGLOBAL::gpu_def[i].gather_seg_index = gp_array[i].gather_seg_index;

		MGLOBAL::gpu_def[i].scatter_seg_index = gp_array[i].scatter_seg_index;


		MGLOBAL::gpu_def[i].gather_table = gp_array[i].gather_table;
		MGLOBAL::gpu_def[i].scatter_table = gp_array[i].scatter_table;

		MGLOBAL::gpu_def[i].gather_table_size = gp_array[i].gather_table_size;
		MGLOBAL::gpu_def[i].scatter_table_size = gp_array[i].scatter_table_size;
		MGLOBAL::total_edge_count[i] = gp_array[i].graph.totalEdgeCount;
		switch (MGLOBAL::graph_ds_type)
		{
		case DS_MEG:
			MGLOBAL::gpu_def[i].edgeArray.buildMEG(gp_array[i].graph);
			break;
		case DS_AA:
			MGLOBAL::gpu_def[i].edgeArray.buildAA(gp_array[i].graph);
			break;
		case DS_HY:
			MGLOBAL::gpu_def[i].edgeArray.buildHY(gp_array[i].graph, MGLOBAL::gpu_def[i].vertexArray,16);
			break;
		default:
			printf("Must define a data type for graph representation!\n");
			exit(-1);
		}
		printf("edge array built for GPU %d\n",i);

	}




	printf("Host data structure created\n");
}


/*
check whether each vertex has outgoing edge

*/
void CheckOutgoingEdge(GraphPartition* gp_array)
{

	//check whether the input graph satisfy the condition for EDGE_MESSAGE (each vertex
	//should have at least one outgoing edge)
	if(MGLOBAL::graph_ds_type == DS_HY)
		return;
	int vcount = 0;
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		bool *vertex_outgoing_flag =(bool*)malloc(sizeof(bool)*gp_array[i].graph.vertexNum);
		memset(vertex_outgoing_flag, 0, sizeof(bool)*gp_array[i].graph.vertexNum);
		for(int j = 0; j < MGLOBAL::gpu_def[i].edgeArray.size; j ++)
		{
			vertex_outgoing_flag[MGLOBAL::gpu_def[i].edgeArray.srcVertexID[j]] = true;

		}
		for(int j = 0; j < gp_array[i].graph.vertexNum; j ++)
		{
			if(!vertex_outgoing_flag[j])
			{
				vcount ++;
			}
		}
	
		free(vertex_outgoing_flag);
	}

	printf("%d vertices have no outgoing edges\n",vcount);

}






/**
Initialize device memory data structures for all devices.

*/
void InitDeviceDS()
{

	MGLOBAL::original_buffer_size = 0;
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		MGLOBAL::original_buffer_size += MGLOBAL::gpu_def[i].gather_table_size;


	//GPU global variables
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);

		//streams
		cudaStreamCreate(&(MGLOBAL::gpu_def[i].async_stream));
		printf("async stream created for GPU %d\n",i);
		cudaStreamCreate(&(MGLOBAL::gpu_def[i].sync_stream));
		printf("sync stream created for GPU %d\n",i);


		//edge and vertex array		
		MGLOBAL::gpu_def[i].d_vertexArray.size = 0;
		MGLOBAL::gpu_def[i].d_edgeArray.size = 0;
		MGLOBAL::gpu_def[i].d_edgeArray.Fill(MGLOBAL::gpu_def[i].edgeArray);
		printf("@@@@edgeArray.size = %d\n", MGLOBAL::gpu_def[i].edgeArray.size);
		MGLOBAL::gpu_def[i].d_vertexArray.Fill(MGLOBAL::gpu_def[i].vertexArray);
		printf("@@@@vertexArray.size = %d\n",MGLOBAL::gpu_def[i].d_vertexArray.size);
		//copy to symbols
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_edgeArray, &MGLOBAL::gpu_def[i].d_edgeArray, sizeof(MGLOBAL::gpu_def[i].d_edgeArray), 0, cudaMemcpyHostToDevice);
		DBGPrintf("1 Symbol copied\n");
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_vertexArray, &MGLOBAL::gpu_def[i].d_vertexArray, sizeof(MGLOBAL::gpu_def[i].d_vertexArray), 0, cudaMemcpyHostToDevice);
		DBGPrintf("Symbol copied\n");

		/*prepare message buffer*/
		int msg_buffer_size;
		if(MGLOBAL::message_mode == EDGE_MESSAGE)
			msg_buffer_size = MGLOBAL::total_edge_count[i];
		else if(MGLOBAL::message_mode == VERTEX_MESSAGE)
			msg_buffer_size = MGLOBAL::gpu_def[i].vertexArray.size;
		else
			msg_buffer_size = 0;

		if(msg_buffer_size != 0)
		{
			MGLOBAL::gpu_def[i].d_messageArray.size = 0;
			MGLOBAL::gpu_def[i].d_messageArray.resize(msg_buffer_size);
			cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_messageArray, &MGLOBAL::gpu_def[i].d_messageArray, sizeof(MGLOBAL::gpu_def[i].d_messageArray),0,cudaMemcpyHostToDevice);
		

			MGLOBAL::gpu_def[i].d_messageArrayBuf.size = 0;
			MGLOBAL::gpu_def[i].d_messageArrayBuf.resize(msg_buffer_size);
			
			cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_messageArrayBuf, &MGLOBAL::gpu_def[i].d_messageArrayBuf, sizeof(MGLOBAL::gpu_def[i].d_messageArrayBuf),0,cudaMemcpyHostToDevice);
			if(MGLOBAL::message_mode == EDGE_MESSAGE)
				MGLOBAL::com.init(MGLOBAL::combiner_datatype, MGLOBAL::combiner_operator, i);//must be initialized after init_Medusa		
		}



		//CPU global variables
		MGLOBAL::super_step = 0;
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_toExecute, &MGLOBAL::toExecute, sizeof(MGLOBAL::toExecute),0,cudaMemcpyHostToDevice);


		//copy gather table

		cudaMalloc((void**)&MGLOBAL::gpu_def[i].d_gather_table, sizeof(int)*MGLOBAL::gpu_def[i].gather_table_size);
		cudaMemcpy(MGLOBAL::gpu_def[i].d_gather_table, MGLOBAL::gpu_def[i].gather_table, sizeof(int)*MGLOBAL::gpu_def[i].gather_table_size, cudaMemcpyHostToDevice);

		//copy scatter table
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].d_scatter_table, sizeof(int)*MGLOBAL::gpu_def[i].scatter_table_size);
		cudaMemcpy(MGLOBAL::gpu_def[i].d_scatter_table, MGLOBAL::gpu_def[i].scatter_table, sizeof(int)*MGLOBAL::gpu_def[i].scatter_table_size, cudaMemcpyHostToDevice);

		//allocate GPU side gather/scatter buffer
		//printf("gather_table_size = %d scatter_table_size = %d\n",MGLOBAL::gpu_def[i].gather_table_size,MGLOBAL::gpu_def[i].scatter_table_size);
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].d_gather_buffer, sizeof(MVT)*MGLOBAL::gpu_def[i].gather_table_size);
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].d_scatter_buffer, sizeof(MVT)*MGLOBAL::gpu_def[i].scatter_table_size);


		free(MGLOBAL::gpu_def[i].gather_table);
		free(MGLOBAL::gpu_def[i].scatter_table);



		//allocate memory for edge and vertex queue
	#if defined EAlpha
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].edgeQueueAlpha, sizeof(int)*MGLOBAL::gpu_def[i].edgeArray.size);
	#endif

	#if defined EBeta
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].edgeQueueBeta, sizeof(int)*MGLOBAL::gpu_def[i].edgeArray.size);
	#endif

	#if defined EAlpha or defined EBeta
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].edgeQueuePtr, sizeof(int));
		SetEdgeQueuePtr(0);
	#endif



	#if defined VAlpha
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].vertexQueueAlpha, sizeof(int)*MGLOBAL::gpu_def[i].vertexArray.size);
	#endif

	#if defined VBeta
	/*
 *		Size of this vetex queue is set to |E|, since the EQ2VQ kernel of SSSP may emit more than |V| vertices.
 *
 */
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].vertexQueueBeta, 2*sizeof(int)*MGLOBAL::gpu_def[i].edgeArray.size);

	#endif

	#if defined VAlpha or defined VBeta
		printf("setup vertex queue ptr\n");
		cudaMalloc((void**)&MGLOBAL::gpu_def[i].vertexQueuePtr, sizeof(int));
		SetVertexQueuePtr(0);
	#endif


		cutilCheckMsg("Global variables");	
	}




	//allocate original buffer
	if(MGLOBAL::num_gpu_to_use > 1)
	{
		printf("original_buffer_size = %d\n",MGLOBAL::original_buffer_size );
		if(cudaMallocHost((void**)&(MGLOBAL::original_buffer), sizeof(MVT)*MGLOBAL::original_buffer_size, cudaHostAllocPortable)!= cudaSuccess)
		{
			printf("cudaMallocHost failed\n");
			exit(-1);
		}


		/*

		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);
			if(cudaMallocHost((void**)&(MGLOBAL::gpu_def[i].replica_buffer), sizeof(MVT)*(MGLOBAL::gpu_def[i].seg_end[2] - MGLOBAL::gpu_def[i].seg_end[1]), cudaHostAllocDefault) != cudaSuccess)
			{
				printf("cudaMallocHost failed\n");
				exit(-1);
			}
		}

		*/
		cutilCheckMsg("allocate original buffer");

	}

	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		cutilCheckMsg("Init Device Data Structure\n");
	}


	printf("Device data structure created\n");


}

void ResetGraphData()
{
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);

		//edge and vertex array		
		MGLOBAL::gpu_def[i].d_vertexArray.size = 0;
		MGLOBAL::gpu_def[i].d_edgeArray.size = 0;
		MGLOBAL::gpu_def[i].d_edgeArray.Fill(MGLOBAL::gpu_def[i].edgeArray);
		MGLOBAL::gpu_def[i].d_vertexArray.Fill(MGLOBAL::gpu_def[i].vertexArray);
		//copy to symbols
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_edgeArray, &MGLOBAL::gpu_def[i].d_edgeArray, sizeof(MGLOBAL::gpu_def[i].d_edgeArray),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_vertexArray, &MGLOBAL::gpu_def[i].d_vertexArray, sizeof(MGLOBAL::gpu_def[i].d_vertexArray),0,cudaMemcpyHostToDevice);
		MGLOBAL::super_step = 0;
	}
}




//---------------------------------------------------------------------------------------------------------//

#include "../Algorithm/InitMessage.h"


/**
* interface function for initialize the message array
*
* @param []	
* @return	
* @note	
*
*/
void InitMessageBuffer(Message init_val)
{
	MyCheckErrorMsg("before MsgArrayInit");

	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		int max_active_warp = 0;
		int major = MGLOBAL::gpu_def[i].device_prop.major;
		int minor = MGLOBAL::gpu_def[i].device_prop.minor;
		if(major >= 2)
			max_active_warp = 48;
		else if(minor >= 2)
			max_active_warp = 32;
		else
			max_active_warp = 24;
		int sm = MGLOBAL::gpu_def[i].device_prop.multiProcessorCount;
		int gridX = 8*sm; //each sm is assigned 8 blocks
		int blockX = (max_active_warp/8)*32;
		int total_thread_count = blockX*gridX;
		MsgArrayInit<<<gridX, blockX>>>(init_val, total_thread_count);
		
	}
	
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		cudaDeviceSynchronize();
		MyCheckErrorMsg("MsgArrayInit");
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_init_msg_val, &init_val, sizeof(init_val), 0,cudaMemcpyHostToDevice);
		MyCheckErrorMsg("After MsgArrayInit");
	}
}

/**
* flush message from d_messageArray to d_messageArrayBuf
*
* @param   - 
* @return	
* @note	
*
*/
void NULL_Combine()
{
	D_MessageArray temp;
	/* swap CPU side pointers */
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		temp = MGLOBAL::gpu_def[i].d_messageArray;
		MGLOBAL::gpu_def[i].d_messageArray = MGLOBAL::gpu_def[i].d_messageArrayBuf;
		MGLOBAL::gpu_def[i].d_messageArrayBuf = temp;

		/* update GPU side variables */
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_messageArray, &(MGLOBAL::gpu_def[i].d_messageArray), sizeof(MGLOBAL::gpu_def[i].d_messageArray),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_messageArrayBuf, &(MGLOBAL::gpu_def[i].d_messageArrayBuf), sizeof(MGLOBAL::gpu_def[i].d_messageArrayBuf),0,cudaMemcpyHostToDevice);
	}
}




#include "../Algorithm/UserOperators.h"



#endif
