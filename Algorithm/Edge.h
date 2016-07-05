#ifndef EDGE_H
#define EDGE_H

/**
* @note user defined data structure for an Edge
* @dev use MACRO to generate its member functions and EdgeArray/D_EdgeArray
*/
struct Edge
{
//---------------------------------------------------------------------------------------------------------//
//	the following attributes are MUST have attributes for the Edge structure, however, users may not need
//	all of them. This should be extracted to be the base class in future versions.
	int srcVertexID;/* the source vertex ID of this edge */
	int dstVertexID;/* the destination vertex ID of this edge */
	int msgDstID;/* this edge send msg to d_messageArray[msgDstID] */
	int weight;/* the weight of this edges */
//---------------------------------------------------------------------------------------------------------//
};

#endif