
#include "GraphGenerator.h"
#include "../Algorithm/EdgeDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../Algorithm/VertexDataType.h"
#include "Utilities.h"
#include "../Algorithm/Configuration.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include <fstream>


/**
* Generate a graph with random number of degrees.
* The degree is between [1, maxDegree]
* @param [in] vertexNum the number of vertexes
* @param [in] maxDegree the maximum degree of all vertexes
* @param [out] graph IR of the generated graph, memory allocated inside the function 
* @note	
*
*/
void generateRandomGraph(const int vertexNum, const int maxDegree, GraphIR &graph)
{
	graph.resize(vertexNum);
	
	for(int i = 0; i < vertexNum; i ++)
	{
		int edgeNum = rand();
		//printf("edgeNum = %d\n",edgeNum);
	//	printf("1 maxDegree = %d\n",maxDegree);
		edgeNum = edgeNum%maxDegree + 1;
		//printf("edgeNum = %d\n",edgeNum);
		//printf("2 maxDegree = %d\n",maxDegree);
		for(int j = 0; j < edgeNum; j++)
		{
			//printf("3 maxDegree = %d\n",maxDegree);
			Edge tempEdge;
#ifndef TEST_DS
			tempEdge.srcVertexID = i;
			//printf("4 maxDegree = %d\n",maxDegree);
			tempEdge.dstVertexID = rand32()%vertexNum;//[0,vertexNum)
			//printf("5 maxDegree = %d\n",maxDegree);
			graph.AddEdge(tempEdge);
			//printf("6 maxDegree = %d\n",maxDegree);
#else
			graph.AddEdge(tempEdge, i);
#endif
		}
	}
	printf("graph generated\n");
}



/**
* Generate a undirected bipartie graph, each set have the same number of vertices; each edge picks up 
* one end points from each the two vertex sets respectively and randomly. Each vertex has 
* equal opportunity to be source vertex of an edge.
* @param [in] vertex_num the number of vertexes
* @param [in] edge_num the number of edges
* @param [out] graph IR of the generated graph, memory allocated inside the function 
* @param [out] set_info indicate which vertex belongs to which set, allocated and set inside this function


	Futuer work: add support for generating multiple partitions

*/
void generateBipartieGraph(const int vertex_num, const int edge_num, GraphPartition *gp_array, int **set_info)
{
	//generate set_info
	if(MGLOBAL::num_gpu_to_use != 1)
	{
		printf("Only support one partition at the moment\n");
		exit(-1);
	}
	*set_info = (int*)malloc(sizeof(int)*vertex_num);
	memset(*set_info, 0, sizeof(int)*vertex_num);
	for(int i = 0; i < vertex_num; i ++)
	{
		(*set_info)[i] = rand()%2;
	}
	printf("set divided\n");
	gp_array[0].graph.resize(vertex_num);
	for(int i = 0; i < edge_num; i ++)
	{
		//pick up a left vertex 0
		int left_id = rand32()%vertex_num;
		while((*set_info)[left_id] == 1)
			left_id = rand32()%vertex_num;
	
		//pick up a right vertex 1
		int right_id = rand32()%vertex_num;
		while((*set_info)[right_id] == 0)
			right_id = rand32()%vertex_num;

		Edge tempEdge;
#ifndef TEST_DS	
		tempEdge.srcVertexID = left_id;
		tempEdge.dstVertexID = right_id;
		gp_array[0].graph.AddEdge(tempEdge);

		tempEdge.srcVertexID = right_id;
		tempEdge.dstVertexID = left_id;
		gp_array[0].graph.AddEdge(tempEdge);
#else
		gp_array[0].graph.AddEdge(tempEdge, left_id);
		gp_array[0].graph.AddEdge(tempEdge, right_id);
#endif

	}
}


void ReadBipartieGraph(char *bm_file_name, char *set_file_name, GraphPartition *gp_array, int **set_info)
{
        //read set_info
        if(MGLOBAL::num_gpu_to_use != 1)
        {
                printf("Only support one partition at the moment\n");
                exit(-1);
        }

	ifstream gt_file(bm_file_name);
	int vertex_num = -1, edge_num = -1;
        char line[1024];
        char first_ch;
        while(gt_file.get(first_ch))
        {
                if(first_ch == 'p')
                {
                        string temp;
                        gt_file>>temp>>vertex_num>>edge_num;
                        gt_file.getline(line, 1024);//eat the line break
                        break;
                }
                gt_file.getline(line, 1024);//eat the line break
        }





        *set_info = (int*)malloc(sizeof(int)*vertex_num);
        memset(*set_info, 0, sizeof(int)*vertex_num);
        ifstream set_file(set_file_name);
        for(int i = 0; i < vertex_num; i ++)
        {
                set_file>>(*set_info)[i];
        }
        printf("set file read\n");


        gp_array[0].graph.resize(vertex_num);
        while(gt_file.get(first_ch))
        {
		if(first_ch == 'a')
		{

	                Edge tempEdge;
			int src_id, dst_id;
			float edge_weight;
			gt_file>>src_id>>dst_id>>edge_weight;
#ifndef TEST_DS
	                tempEdge.srcVertexID = src_id;
	                tempEdge.dstVertexID = dst_id;
	                gp_array[0].graph.AddEdge(tempEdge);

        	        tempEdge.srcVertexID = dst_id;
       	       		tempEdge.dstVertexID = src_id;
                	gp_array[0].graph.AddEdge(tempEdge);
#else
                	gp_array[0].graph.AddEdge(tempEdge, src_id);
                	gp_array[0].graph.AddEdge(tempEdge, dst_id);
#endif
		}

        }
}



void generateZipfGraph(const int vertex_num, const int edge_num, double skew, bool directed, GraphIR &graph)
{
	/* generate zipf distribution */
	int length = vertex_num;
	double factor = edge_num;
	int *data = (int*)malloc(sizeof(int)*vertex_num);
	double sum = 0;
	for(int i = 0;i< length; i++)
		sum += 1/pow((double)(i+1), skew);
	for(int i = 0; i < length; i++)
	{
		double temp = 1.0/pow(double(i+1), skew)/sum*factor;
		data[i] = temp;
		if(temp - data[i] >= 0.5)
			data[i] ++;
	}

	/* generate the graph based on data[] */
	for(int i = 0; i < vertex_num; i ++)
	{
		for(int j = 0; j < data[i]; j++)
		{
			Edge tempEdge;
			tempEdge.srcVertexID = i;
			tempEdge.dstVertexID = rand32()%vertex_num;//[0,vertexNum)
			graph.AddEdge(tempEdge);
		}
	}
}
