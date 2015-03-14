#ifndef USEROPERATORS_H
#define USEROPERATORS_H

//-------------------Medusa API Implementation------------------------------------------------------//

/**
* this function will run inside the BSP loop
* firstly user should define the functors (functor types: EDGE,EDGELIST,VERTEX,MESSAGE,MESSAGELIST)
* Then functos are passed to functor holders and user should indicate the functor type
* @param []	
* @return	
* @note	
*
*/

struct UpdateVertex
{
	__device__ void operator() (D_Vertex vertex, int super_step)
	{
		MVT msg_sum = vertex.get_combined_msg();
		//printf("[%d]%f\n",vertex.index, msg_sum);
		msg_sum *= 0.85;
		vertex.set_level(0.15 + msg_sum);
		//printf("%d\n", vertex.get_edge_count());
	}
};


struct SendMsg
{
	__device__ void operator() (D_Edge e, int super_step)
	{
		int src = e.get_srcVertexID();
		D_Vertex srcV(src);
		//printf("src = %d\n",src);
		//	printf("edge_count = %d\n",srcV.get_edge_count());
		e.sendMsg(srcV.get_level()/srcV.get_edge_count());
		//printf("%f/%d = %f\n",srcV.get_level(),srcV.get_edge_count(), srcV.get_level()/srcV.get_edge_count());

		//e.sendMsg(srcV.get_level());
		//e.sendMsg(srcV.get_edge_count());
	}
};


//-------------------------------Testing Different Edge Representations----------------------------//
#ifdef AA
__global__ void SendMsgAA(int super_step, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for( ; tid < GRAPH_STORAGE_GPU::d_vertexArray.size; tid += total_thread_num)
	{
		//calculate message
		MVT level = GRAPH_STORAGE_GPU::d_vertexArray.d_level[tid];
		int edge_count = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_count[tid];
		MVT msg = level/edge_count;

		int start_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid];
		int end_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid + 1];
		//printf("start %d end %d\n",start_index, end_index);
		for(; start_index < end_index; start_index ++)
		{

			//send message
			int msg_dst_index = GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[start_index];
			GRAPH_STORAGE_GPU::d_messageArray.d_val[msg_dst_index] = msg;
			//printf("[%d]=%f\n",msg_dst_index,msg);
		}
	}
}
#endif


#ifdef HY
__global__ void SendMsgHY(int super_step, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	for( ; tid < GRAPH_STORAGE_GPU::d_vertexArray.size; tid += total_thread_num)
	{
	
		int edge_count = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_count[tid];
		int start_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid];
		int end_index = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_index[tid + 1];
		//calculate message
		MVT level = GRAPH_STORAGE_GPU::d_vertexArray.d_level[tid];
		MVT msg = level/(edge_count + end_index - start_index);


		int stride = GRAPH_STORAGE_GPU::d_vertexArray.size;
		int fetch = tid;
		//printf("fetch = %d msg = %f/(%d + %d - %d) \n",tid,level, edge_count, end_index ,start_index);

		for(int i = 0; i < edge_count; i ++)
		{
			//send message
			int msg_dst_index = GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[fetch];
		//	printf("msg_dst_index = %d\n",msg_dst_index);
		//	if(msg_dst_index  >= GRAPH_STORAGE_GPU::d_messageArray.size)
		//		printf("msg_dst_index = %d\n",msg_dst_index);
			GRAPH_STORAGE_GPU::d_messageArray.d_val[msg_dst_index] = msg;
			fetch += stride;
		}


		for(; start_index < end_index; start_index ++)
		{
			//send message
			int msg_dst_index = GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[start_index];
			GRAPH_STORAGE_GPU::d_messageArray.d_val[msg_dst_index] = msg;

		}
	}
}
#endif


#ifdef MEG
__global__ void SendMsgMEG(int super_step, int total_thread_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	for( ; tid < GRAPH_STORAGE_GPU::d_vertexArray.size; tid += total_thread_num)
	{


		//calculate message
		MVT level = GRAPH_STORAGE_GPU::d_vertexArray.d_level[tid];
		int edge_count = GRAPH_STORAGE_GPU::d_vertexArray.d_edge_count[tid];
		MVT msg = level/edge_count;


		//
		////printf("GRAPH_STORAGE_GPU::d_edgeArray.size = %d edge_count = %d\n",GRAPH_STORAGE_GPU::d_edgeArray.size,edge_count );
		int fetch = tid;
		for(int i = 0; i < edge_count; i ++)
		{
			int msg_dst_index = GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[fetch];
			//printf("fetch = %d msg_dst_index = %d\n",fetch, msg_dst_index);
			GRAPH_STORAGE_GPU::d_messageArray.d_val[msg_dst_index] = msg;		
			fetch += GRAPH_STORAGE_GPU::d_edgeArray.d_edgeOffset[i];
		}
	}

}
#endif



void SM_EdgeList_Processor()
{

	int gridX = MGLOBAL::gpu_def[0].device_prop.multiProcessorCount*6;
#if defined(AA)
	SendMsgAA<<<gridX, 256>>>(MGLOBAL::super_step, gridX*256);
#elif defined(HY)
	SendMsgHY<<<gridX, 256>>>(MGLOBAL::super_step, gridX*256);
#elif defined(MEG)
	SendMsgMEG<<<gridX, 256>>>(MGLOBAL::super_step, gridX*256);
#endif
	cudaDeviceSynchronize();
	MyCheckErrorMsg("after edge list");

}



//---------------------------------------------------------------------------------------------------------//
SendMsg sm;
UpdateVertex uv;
Message init_msg;

unsigned int init_time, send_time, com_time, update_time;

#define TIMEING_EACH_OPERATOR
void Medusa_Exec()
{
#ifdef TIMEING_EACH_OPERATOR


	if(MGLOBAL::super_step == 0)	
	{
		printf("Timer created\n");
		cutCreateTimer(&init_time);
		cutCreateTimer(&com_time);
		cutCreateTimer(&send_time);
		cutCreateTimer(&update_time);
		cutResetTimer(init_time);
		cutResetTimer(com_time);
		cutResetTimer(send_time);
		cutResetTimer(update_time);
	}
#endif


#ifdef TIMEING_EACH_OPERATOR
	cutStartTimer(init_time);
#endif

	init_msg.val = 0.0;
	InitMessageBuffer(init_msg);
#ifdef TIMEING_EACH_OPERATOR
	cutStopTimer(init_time);
#endif


#ifdef TIMEING_EACH_OPERATOR
	cutStartTimer(send_time);
#endif
	if(MGLOBAL::num_gpu_to_use > 1  && MGLOBAL::max_hop >= 1)
	{
		MGFunctorHolder<MGMH_EDGE>::Run(sm);
	}
	else if(MGLOBAL::num_gpu_to_use > 1)
	{
		MGFunctorHolder<MG_EDGE>::Run(sm);
		//SM_EdgeList_Processor();
	}
	else
	{
		//FunctorHolder<EDGE>::Run(sm);
		SM_EdgeList_Processor();
	}
#ifdef TIMEING_EACH_OPERATOR
	cutStopTimer(send_time);
#endif

#ifdef TIMEING_EACH_OPERATOR
	cutStartTimer(com_time);
#endif

	//combiner
	MGLOBAL::com.combineAllDevice();
#ifdef TIMEING_EACH_OPERATOR
	cutStopTimer(com_time);
#endif


#ifdef TIMEING_EACH_OPERATOR
	cutStartTimer(update_time);
#endif

	if(MGLOBAL::num_gpu_to_use > 1 && MGLOBAL::max_hop >= 1)//&& MGLOBAL::max_hop > 1)
	{
		MGFunctorHolder<MGMH_VERTEX>::Run(uv);
	}
	else if(MGLOBAL::num_gpu_to_use > 1)
		MGFunctorHolder<MG_VERTEX>::Run(uv);
	else
		FunctorHolder<VERTEX>::Run(uv);

#ifdef TIMEING_EACH_OPERATOR
	cutStopTimer(update_time);
#endif


#ifdef TIMEING_EACH_OPERATOR
	if(MGLOBAL::super_step == 99)
	{
		float duration;
		duration = cutGetTimerValue(init_time);
		printf("Init time %f ms\n", duration);
		duration = cutGetTimerValue(send_time);
		printf("Send time %f ms\n", duration);
		duration = cutGetTimerValue(com_time);
		printf("Com time %f ms\n", duration);
		duration = cutGetTimerValue(update_time);
		printf("Update time %f ms\n", duration);
	}
#endif
}


#endif
