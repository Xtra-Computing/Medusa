#ifndef GRAPHGENERATOR_H
#define GRAPHGENERATOR_H
#include "GraphConverter.h"
#include "../MultipleGPU/PartitionManager.h"

/**
* Generate a graph with random number of degrees.
* The degree is between [1, maxDegree]
* @param [in] vertexNum the number of vertexes
* @param [in] maxDegree the maximum degree of all vertexes
* @param [out] graph IR of the generated graph, memory allocated inside the function 
* @note	
*
*/
void generateRandomGraph(const int vertexNum, const int maxDegree, GraphIR &graph);


/**
* Generate a bipartie graph, each set have the same number of vertices; each edge picks up 
* one end points from each the two vertex sets respectively and randomly
* @param [in] vertex_num the number of vertexes
* @param [in] edge_num the number of edges
* @param [out] graph IR of the generated graph, memory allocated inside the function 
* @param [out] set_info indicate which vertex belongs to which set, allocated and set inside this function

*/
void generateBipartieGraph(const int vertex_num, const int edge_num, GraphPartition *gp_array, int **set_info);


void ReadBipartieGraph(char *bm_file_name, char *set_file_name, GraphPartition *gp_array, int **set_info);

/**
*
*
* @param skew - skew of zipf
* @param directed - whether i the graph i directed. if undirected each edge nees
					to be added twice.
* @return	
* @note	
*
*/
void generateZipfGraph(const int vertex_num, const int edge_num, double skew, bool directed, GraphIR &graph);

#endif
