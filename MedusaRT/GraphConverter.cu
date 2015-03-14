/*
*/
#include "GraphConverter.h"
#include <algorithm>
#include "../MedusaRT/Utilities.h"
#include "../Algorithm/EdgeDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../Algorithm/VertexDataType.h"

#include "../Algorithm/Configuration.h"


/*
* the constructor of GraphIR
* @param [in] vertexNum the number of vertices of the input graph
*/
GraphIR::GraphIR(int num)
{
	if(num <= 0)
	{
		printf("num should be larger than 0\n");
		exit(-1);
	}
	vertexNum = num;
	vertexArray = (VertexNode*) malloc(sizeof(VertexNode)*num);
	L2P = (int*) malloc(sizeof(int)*vertexNum);
	/* initialize the empty VertexNode array */
	for(int i = 0; i < num; i ++)
	{
		vertexArray[i].firstEdge = NULL;
		vertexArray[i].vertex.edge_count = 0;
		vertexArray[i].logicalID = i;
		L2P[i] = i;
	}
}

/**
* default constructor
*/
GraphIR::GraphIR()
{
	vertexNum = 0;
	totalEdgeCount = 0;
	vertexArray = NULL;
	L2P =NULL;
}

/*
*  resize the size of this IR, erase existing data
* @param [in] vertexNum the number of vertices of the input graph
*/
void GraphIR::resize(int num)
{
	if(vertexNum > 0)
	{
		free(vertexArray);
		free(L2P);
		totalEdgeCount = 0;
	}
	if(num <= 0)
	{
		printf("num should be larger than 0\n");
		exit(-1);
	}
	vertexNum = num;
	CPUMalloc((void**)&vertexArray, sizeof(VertexNode)*num);
	CPUMalloc((void**)&L2P, sizeof(int)*vertexNum);
	/* initialize the empty VertexNode array */
	for(int i = 0; i < num; i ++)
	{
		vertexArray[i].firstEdge = NULL;
		vertexArray[i].vertex.edge_count = 0;
		vertexArray[i].incoming_edge_count = 0;
		vertexArray[i].logicalID = i;
		L2P[i] = i;
	}
}




/*
* add one edge to the graph IR
* @param [in] the edge object to be added
*/
#ifndef TEST_DS
void GraphIR::AddEdge(Edge edge)
{

	int srcID = edge.srcVertexID;

	EdgeNode *newEdgeNode;
	CPUMalloc((void**)&newEdgeNode, sizeof(EdgeNode));

	/* this should be a deep copy if there is any reference types in Edge */
	newEdgeNode->edge = edge;
	//printf("src id = %d\n",srcID);
	newEdgeNode->nextEdge = vertexArray[srcID].firstEdge;
	vertexArray[srcID].firstEdge = newEdgeNode;
	vertexArray[srcID].vertex.edge_count ++;
	vertexArray[edge.dstVertexID].incoming_edge_count ++;
	totalEdgeCount ++;
}
#else
/*
* add one edge to the graph IR
* @param [in] the edge object to be added
*/
void GraphIR::AddEdge(Edge edge, int srcID)
{

	EdgeNode *newEdgeNode;
	CPUMalloc((void**)&newEdgeNode, sizeof(EdgeNode));

	/* this should be a deep copy if there is any reference types in Edge */
	newEdgeNode->edge = edge;
	//printf("src id = %d\n",srcID);
	newEdgeNode->nextEdge = vertexArray[srcID].firstEdge;
	vertexArray[srcID].firstEdge = newEdgeNode;
	vertexArray[srcID].vertex.edge_count ++;
	totalEdgeCount ++;
}
#endif

bool cmpVertexNode(VertexNode l, VertexNode r)
{
	return (l.vertex.edge_count > r.vertex.edge_count);
}

/**
* sort the graph by degree and generate the logical to physical map(L2P)
* then adjust the graph according to the logical to physical map
* at last, calculate the index for retrieving messages sent along edges
*/

void GraphIR::sortByDegree(bool (*cmp)(VertexNode, VertexNode))
{
	std::sort(vertexArray, &vertexArray[vertexNum], cmp);
	//* Generate Logical To Physical Map */
	for(int i = 0; i < vertexNum; i ++)
	{
		L2P[vertexArray[i].logicalID] = i;
	}

	/* adjust graph  */
	//(the edge dst/src should be in accordance with the vertex physical ID)
	for(int i = 0; i < vertexNum; i ++)
	{
		// loop through the edge list
		if(vertexArray[i].vertex.edge_count)
		{
			EdgeNode *tempEdgeNode = vertexArray[i].firstEdge;
			while(tempEdgeNode != NULL)
			{
#ifndef TEST_DS
				tempEdgeNode->edge.dstVertexID = L2P[tempEdgeNode->edge.dstVertexID];
				tempEdgeNode->edge.srcVertexID = L2P[tempEdgeNode->edge.srcVertexID];
#endif
				tempEdgeNode = tempEdgeNode->nextEdge;
			}
		}
	}


}

