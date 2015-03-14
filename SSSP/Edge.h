/**
* @note user defined data structure for an Edge
* @dev use MACRO to generate its member functions and EdgeArray/D_EdgeArray
*/
struct Edge
{
	int srcVertexID;/* the source vertex ID of this edge */
	int dstVertexID;/* the destination vertex ID of this edge */
	int msgDstID;/* this edge send msg to d_messageArray[msgDstID] */
	MVT weight;
};

