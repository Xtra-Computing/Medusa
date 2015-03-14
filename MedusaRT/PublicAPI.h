/****************************************************
* @file 
* @brief users write their application logic here
* @version
* @author Zhong Jianlong(http://www.jlzhong.com)
* @date 2011/01/03
* Copyleft for non-commercial use only. No warranty.
****************************************************/
#ifndef PUBLICAPI_H
#define PUBLICAPI_H

#include "Utilities.h"
#include "APIHosters.h"
#include "../Algorithm/EdgeDataType.h"
#include "../Algorithm/VertexDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../Algorithm/Configuration.h"
#include "../MedusaRT/CUDAOpenglInterop.h"

void Medusa_Init_Config(bool loop_behavior, MessageMode msg_mode)
{
	GRAPH_STORAGE_CPU::message_mode = msg_mode;
	GRAPH_STORAGE_CPU::toExecute = loop_behavior;
}
/**
*
*
* @param []	
* @return	
* @note	
*
*/

void Medusa_Init_Data(EdgeArray ea, VertexArray varr, D_EdgeArray d_ea, D_VertexArray d_varr
#ifdef HY
	,int actual_message_buf_size
#endif
					  )
{
	//GPU global variables

	cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_edgeArray, &d_ea, sizeof(d_ea),0,cudaMemcpyHostToDevice);
	GRAPH_STORAGE_CPU::alias_d_edgeArray = d_ea;
	cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_vertexArray, &d_varr, sizeof(d_varr),0,cudaMemcpyHostToDevice);
	GRAPH_STORAGE_CPU::alias_d_vertexArray = d_varr;
	

	/* prepare message buffer */
	int msg_buffer_size;
	if(GRAPH_STORAGE_CPU::message_mode == EDGE_MESSAGE)
		msg_buffer_size = ea.size;
	else if(GRAPH_STORAGE_CPU::message_mode == VERTEX_MESSAGE)
		msg_buffer_size = varr.size;
	else
		msg_buffer_size = 0;
#ifdef HY
	msg_buffer_size = actual_message_buf_size;
#endif
	//printf("msg_buffer_size = %d\n",msg_buffer_size);
	if(msg_buffer_size)
	{
		D_MessageArray d_ma;
		d_ma.size = 0;
		d_ma.resize(msg_buffer_size);
		GRAPH_STORAGE_CPU::alias_d_messageArray = d_ma;
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_messageArray, &d_ma, sizeof(d_ma),0,cudaMemcpyHostToDevice);

		D_MessageArray d_mabuf;
		d_mabuf.size = 0;
		d_mabuf.resize(msg_buffer_size);

		GRAPH_STORAGE_CPU::alias_d_messageArrayBuf = d_mabuf;
		cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_messageArrayBuf, &d_mabuf, sizeof(d_mabuf),0,cudaMemcpyHostToDevice);
		MyCheckErrorMsg("before init");
	}
	//CPU global variables
	GRAPH_STORAGE_CPU::edgeArray = ea;
	GRAPH_STORAGE_CPU::vertexArray = varr;
	GRAPH_STORAGE_CPU::super_step = 0;


	cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_toExecute, &GRAPH_STORAGE_CPU::toExecute, sizeof(GRAPH_STORAGE_CPU::toExecute),0,cudaMemcpyHostToDevice);
	
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
	int sm = GRAPH_STORAGE_CPU::device_prop.multiProcessorCount;
	int gridX = 6*sm; //each sm is assigned 8 blocks
	int total_thread_count = gridX*256;
	MsgArrayInit<<<gridX, 256>>>(init_val, total_thread_count);
	cudaThreadSynchronize();
	MyCheckErrorMsg("MsgArrayInit");
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_init_msg_val, &init_val, sizeof(init_val), 0,cudaMemcpyHostToDevice));
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
	temp = GRAPH_STORAGE_CPU::alias_d_messageArray;
	GRAPH_STORAGE_CPU::alias_d_messageArray = GRAPH_STORAGE_CPU::alias_d_messageArrayBuf;
	GRAPH_STORAGE_CPU::alias_d_messageArrayBuf = temp;

	/* update GPU side variables */
	cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_messageArray, &(GRAPH_STORAGE_CPU::alias_d_messageArray), sizeof(GRAPH_STORAGE_CPU::alias_d_messageArray),0,cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(GRAPH_STORAGE_GPU::d_messageArrayBuf, &(GRAPH_STORAGE_CPU::alias_d_messageArrayBuf), sizeof(GRAPH_STORAGE_CPU::alias_d_messageArrayBuf),0,cudaMemcpyHostToDevice);

}


/**
* Printf GRAPH_STORAGE_CPU::d_messageArrayBuf
*/
//void PrintMessageBuffer()
//{
//	MVT *temp = (MVT *)malloc(GRAPH_STORAGE_CPU::alias_d_messageArray.size*sizeof(int));
//	cudaMemcpy(temp, GRAPH_STORAGE_CPU::alias_d_messageArray.d_val_x, GRAPH_STORAGE_CPU::alias_d_messageArray.size*sizeof(MVT), cudaMemcpyDeviceToHost);
//	for(int i = 0; i < GRAPH_STORAGE_CPU::alias_d_messageArray.size; i ++)
//		printf("m[%d]=%f ",i, temp[i]);
//	printf("\n");
//}


#include "../Algorithm/UserOperators.h"



#endif
