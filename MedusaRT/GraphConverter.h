#ifndef GRAPHCONVERTER_H
#define GRAPHCONVERTER_H

#include "../Algorithm/Vertex.h"
#include "../Algorithm/Edge.h"
#include "../Algorithm/Configuration.h"
#ifdef _WIN32
#include <hash_map>
#define HASH_SPACE stdext
#endif

#ifdef __linux__
#include <ext/hash_map>
#define HASH_SPACE __gnu_cxx
#endif
/** intermediate representation of edge nodes */
struct EdgeNode
{
	Edge edge;
	EdgeNode *nextEdge;
};

/** intermediate representation of vertex nodes */
struct VertexNode
{
	Vertex vertex;
	int incoming_edge_count;
	int logicalID;/**< the vertex ID read from file */
	EdgeNode *firstEdge;
};

bool cmpVertexNode(VertexNode l, VertexNode r);

/**
* intermediate representation of graph 
*/
struct GraphIR
{
	VertexNode *vertexArray;/**< the vertex array */
	int *L2P;/**< map from logic ID(read from file) to physical ID(the index in vertexArray[]) */
	HASH_SPACE::hash_map<int, int> hash_L2P;
	int vertexNum;/**<  the number of vertices*/
	int totalEdgeCount;/* the total number of edges */
	/*
	* the constructor of GraphIR
	* @param [in] num the number of vertices of the input graph
	*/
	GraphIR(int num);

	/**
	* the default constructor of GraphIR
	*/
	GraphIR();

	/**
	* resize the size of this IR, erase existing data
	* @param [in] num the number of vertices of the input graph
	* 
	*/
	void resize(int num);

	
#ifndef TEST_DS

	/*
	* add one edge to the graph IR
	* this is the *only* way to add edges to the graph
	* @param [in] the edge object to be added
	*/
	void AddEdge(Edge edge);

#else	
	/*
	* add one edge to the graph IR
	* this is the *only* way to add edges to the graph
	* @param [in] edge -- the edge object to be added
	* @param [in] srcID -- source vertex id of the edge to be added
	*/
	void AddEdge(Edge edge, int srcID);

#endif

	/**
	* sort the graph by degree
	*/
	void sortByDegree(bool (*cmp)(VertexNode, VertexNode) = cmpVertexNode);


};



#endif
