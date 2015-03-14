#ifndef EDGEDATATYPE_H
#define EDGEDATATYPE_H
#include <cutil.h>
#include <cuda_runtime.h>
#include "../Algorithm/EdgeDataType.h"
#include "../Algorithm/VertexDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../MedusaRT/GraphConverter.h"


/**
* @dev under development, should be atomaticaly generated
*/
struct EdgeArray
{
	int *srcVertexID;
	int *dstVertexID;
	int *msgDstID;/* the message position index */
	int *edgeOffset;/* For MEG, the number of edges of each level */
	unsigned int *incoming_msg_flag;

	int level_count; //the number of levels
	int size;
	EdgeArray();
	void resize(int num);
	void assign(int i, Edge e);/* assign Edge e to the element i of this array */
	/**
	* construct the MEG representation of graph
	*/
	void buildMEG(GraphIR &graph);
	/**
	* construct the AA representation of graph
	*/
	void buildAA(GraphIR &graph);
	
	/**
	* construct the ELL representation of the graph
	*/
	void buildELL(GraphIR &graph);

	/**
	* build hybrid (ELL + AA) representation of the graph
	* need to alter the edge_index attribute of the associated vertex array
	*/
	void buildHY(GraphIR &graph, VertexArray &varr, int threshold);
};

/**
* @dev under development, should be atomaticaly generated
*/
struct D_EdgeArray
{
	int *d_srcVertexID;
	int *d_dstVertexID;
	int *d_msgDstID;/* the message position index */
	int *d_edgeOffset;
	unsigned int *d_incoming_msg_flag;
	int size;
	void Fill(EdgeArray &ea);
};


#endif