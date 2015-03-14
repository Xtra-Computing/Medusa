#ifndef DEVICEDATASTRUCTURE_H
#define DEVICEDATASTRUCTURE_H

namespace GRAPH_STORAGE_GPU
{
	//GPU can only referenced from kernels
	__constant__ D_VertexArray d_vertexArray;
	__constant__ int d_test;
	__constant__ D_EdgeArray d_edgeArray;
	__constant__ D_MessageArray d_messageArray;
	__device__ D_MessageArray d_messageArrayBuf;
	__device__ bool d_toExecute;/* termination condition of the BSP iteration */
	/* loop continue to execute or stop without user alter */
	__device__ MVT d_init_msg_val;
}






//---------------------------------------------------------------------------------------------------------//
#include "EdgeDataType.h"
#include "MessageDataType.h"
#include "VertexDataType.h"
#include "Configuration.h"


struct D_Message
{
	int index;
	__device__ D_Message(int index);
	__device__ void set_val(MVT newdistance);
	__device__ MVT get_val();
};


__device__ D_Message::D_Message(int init_index)
{
	index = init_index;
}

__device__ void D_Message::set_val(MVT new_val)
{
	GRAPH_STORAGE_GPU::d_messageArray.d_val[index] = new_val;
}

__device__ MVT D_Message::get_val()
{
	return GRAPH_STORAGE_GPU::d_messageArray.d_val[index];
}

//---------------------------------------------------------------------------------------------------------//

struct D_Edge
{
	int edge_index;
	__device__ D_Edge(int index);/* constructor, used by RT*/
	__device__ void set_srcVertexID(int);
	__device__ int get_srcVertexID();
	__device__ void set_dstVertexID(int);
	__device__ int get_dstVertexID();
	__device__ void sendMsg(MVT v); /* send a message along this edge */
	__device__ MVT get_weight();
	__device__ void set_weight(MVT l);
};



struct EdgeList
{
	int vertex_index;/* physical index of the corresponding vertex of this edge list */
	int current_index;/* the index of the current edge in the GPU Graph Storage */
	int edge_number;/* the current edge is the edge_number th edge of this edge list */
	__device__ EdgeList(int v_index);/* construct the virtual edge */
	__device__ D_Edge getNextEdge();/* get next edge */
	__device__ D_Edge getPreviousEdge();/* get previous edge */
	__device__ int getEdgeCount();/* the number of edges of this edge list */
};


__device__ D_Edge::D_Edge(int index)
{
	edge_index = index;
}

__device__ void D_Edge::set_srcVertexID(int new_src)
{
	GRAPH_STORAGE_GPU::d_edgeArray.d_srcVertexID[edge_index] = new_src;
}

__device__ int D_Edge::get_srcVertexID()
{
	return GRAPH_STORAGE_GPU::d_edgeArray.d_srcVertexID[edge_index];
}

__device__ void D_Edge::set_dstVertexID(int new_dst)
{
	GRAPH_STORAGE_GPU::d_edgeArray.d_dstVertexID[edge_index] = new_dst;
}

__device__ int D_Edge::get_dstVertexID()
{
	return GRAPH_STORAGE_GPU::d_edgeArray.d_dstVertexID[edge_index];
}

__device__ void D_Edge::sendMsg(MVT v)
{
	int msg_dst_index = GRAPH_STORAGE_GPU::d_edgeArray.d_msgDstID[edge_index];
	//printf("msg_dst_index = %d\n",msg_dst_index);
	GRAPH_STORAGE_GPU::d_messageArray.d_val[msg_dst_index] = v;
}

__device__ void D_Edge::set_weight(MVT l)
{
	GRAPH_STORAGE_GPU::d_edgeArray.d_weight[edge_index] = l;
}

__device__ MVT D_Edge::get_weight()
{
	//printf("return edge[%d] weight %f\n",edge_index,GRAPH_STORAGE_GPU::d_edgeArray.d_weight[edge_index]);
	return	GRAPH_STORAGE_GPU::d_edgeArray.d_weight[edge_index];
}


__device__ EdgeList::EdgeList(int v_index)
{
	vertex_index = current_index = v_index;
	edge_number = 0;
}

__device__ D_Edge EdgeList::getNextEdge()
{
	current_index += GRAPH_STORAGE_GPU::d_edgeArray.d_edgeOffset[edge_number++];
	return D_Edge(current_index);
}

__device__ D_Edge EdgeList::getPreviousEdge()
{
	current_index -= GRAPH_STORAGE_GPU::d_edgeArray.d_edgeOffset[--edge_number];
	return D_Edge(current_index);
}

/**
*
* return the number of edges in this edge_list
* @param []	
* @return	
* @note	
*
*/
__device__ int EdgeList::getEdgeCount()
{
	return GRAPH_STORAGE_GPU::d_vertexArray.d_edge_count[vertex_index];
}


//---------------------------------------------------------------------------------------------------------//

struct D_Vertex
{
	int index;
	/**
	* Construct the virtual vertex by the physical id(array index) of the VertexArray
	*
	* @param [in] index physical ID of the vertex
	*
	*/
	__device__ D_Vertex(int index);
	__device__ MVT get_distance();
	__device__ void set_distance(MVT new_distance);
	__device__ int get_msg_index();
	__device__ void set_msg_index(int new_msg_index);
	__device__ MVT get_combined_msg();
	__device__ int get_edge_count();
	__device__ void set_updated(bool up);
	__device__ bool get_updated();


};

__device__ int D_Vertex::get_edge_count()
{
	return GRAPH_STORAGE_GPU::d_vertexArray.d_edge_count[index];
}

__device__ D_Vertex::D_Vertex(int vertex_index)
{
	index = vertex_index;
}

__device__ MVT D_Vertex::get_distance()
{
	return GRAPH_STORAGE_GPU::d_vertexArray.d_distance[index];
}

__device__ void D_Vertex::set_distance(MVT new_distance)
{
	GRAPH_STORAGE_GPU::d_vertexArray.d_distance[index] = new_distance;
}


__device__ int D_Vertex::get_msg_index()
{
	return GRAPH_STORAGE_GPU::d_vertexArray.d_msg_index[index];
}

__device__ void D_Vertex::set_msg_index(int new_index)
{
	GRAPH_STORAGE_GPU::d_vertexArray.d_msg_index[index] = new_index;
}

/**
*
*
* @param []	
* @return	
* @note	!only use it in EDGE_MESSAGE mode

*/
__device__ MVT D_Vertex::get_combined_msg()
{
	
	/*printf("combined position = %d\n",GRAPH_STORAGE_GPU::d_vertexArray.d_msg_index[index + 1]);
	printf("combined msg = %f\n",GRAPH_STORAGE_GPU::d_messageArrayBuf[GRAPH_STORAGE_GPU::d_vertexArray.d_msg_index[index + 1] - 1]);*/
	int next_msg_index = GRAPH_STORAGE_GPU::d_vertexArray.d_msg_index[index + 1];
	int msg_index = GRAPH_STORAGE_GPU::d_vertexArray.d_msg_index[index];
	if(next_msg_index == msg_index)
	{
		//printf("vertex %d has no incoming edges\n",index);
		return GRAPH_STORAGE_GPU::d_init_msg_val;
	}
	else
		return GRAPH_STORAGE_GPU::d_messageArrayBuf.d_val[next_msg_index - 1];
}


__device__ void D_Vertex::set_updated(bool up)
{
	GRAPH_STORAGE_GPU::d_vertexArray.d_updated[index] = up;
}

__device__ bool D_Vertex::get_updated()
{
	return GRAPH_STORAGE_GPU::d_vertexArray.d_updated[index];
}



#endif
