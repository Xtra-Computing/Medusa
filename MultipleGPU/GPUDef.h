#ifndef GPUDEF_H
#define GPUDEF_H

#include "../Algorithm/VertexDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../Algorithm/EdgeDataType.h"
#include "../MedusaRT/Combiner.h"
#include "../Algorithm/Configuration.h"



struct GPUDef
{
	//CPU
	VertexArray vertexArray;
	EdgeArray edgeArray;


	//device variable alias, referenced from Host code
	D_MessageArray d_messageArray;
	D_MessageArray d_messageArrayBuf;
	D_EdgeArray d_edgeArray;
	D_VertexArray d_vertexArray;


	//const MessageArray messageArray;
	cudaDeviceProp device_prop;


	//streams to manage asynchronous execution
	cudaStream_t sync_stream;
	cudaStream_t async_stream;


	//replica buffer (generate from the updated original data)
	int *gather_table;
	int *d_gather_table; //size equals the number of times that vertices in this partition worked as replica in other partitions
	MVT *d_gather_buffer;
	int gather_table_size;

	int *scatter_table;
	int *d_scatter_table; // size equals the number of replicas in this partition
	MVT *d_scatter_buffer;
	int scatter_table_size;

	int *gather_seg_index;
	int *scatter_seg_index;
	

	//for multi-hop
	int *replica_index; //size equals (max_hop + 1)
	int *replica_edge_index; //size equals (max_hop + 1)

	//for cudpp segmented scan
	CUDPPResult cudpp_result;
	CUDPPConfiguration config;
	CUDPPHandle scanplan;
#ifdef CUDPP_2_0
	CUDPPHandle theCudpp;
#endif


	//queue related data structures
	int *edgeQueuePtr;
	void SetEdgeQueueSize(int _size);
	int GetEdgeQueueSize();
	int *vertexQueuePtr;
	void SetVertexQueueSize(int _size);
	int GetVetexQueueSize();
	int *edgeQueueAlpha;
	int *edgeQueueBeta;
	int *vertexQueueAlpha;
	int *vertexQueueBeta;

};




#endif
