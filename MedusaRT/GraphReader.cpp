/****************************************************
* @file 
* @brief 
* @version
* @author Zhong Jianlong(http://www.jlzhong.com)
* @date 2011/01/04
* Copyleft for non-commercial use only. No warranty.
****************************************************/
#include "GraphConverter.h"
#include "GraphReader.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include "../Algorithm/Configuration.h"
#include "../MedusaRT/Utilities.h"

#ifndef TEST_DS

/**
* default constructor of the base class
*
* @param   - 
* @return	
* @note	
*
*/

GraphReader::GraphReader()
{
	printf("get a reader\n");
}

void GraphReader::ReadGraph(string str)
{
	printf("dumb reader\n");
}

/**
*	Read DBLP graph from file
* File Format:

p sp node_count edge_count
a src_node_id dst_node_id number_of_coauthored_paper
a src_node_id dst_node_id number_of_coauthored_paper
a src_node_id dst_node_id number_of_coauthored_paper
a src_node_id dst_node_id number_of_coauthored_paper
...
...
...
a src_node_id dst_node_id number_of_coauthored_paper

* @param file_name  - DBLP graph name
* @return	
* @note	
*
*/

void DBLPReader::ReadGraph(string file_name)
{
	//read the first line
	ifstream fin(file_name.c_str());
	string temp_str;
	char ch;
	while(fin.get(ch))
	{
		if(ch!='p')
		{
			getline(fin,temp_str);
			continue;
		}
		fin>>temp_str;
		fin>>graph.vertexNum>>graph.totalEdgeCount;
		getline(fin,temp_str);
		break;
	}


	//read all the edges
	Edge temp_edge;
	while(fin.get(ch))
	{
		fin>>temp_edge.srcVertexID;
		fin>>temp_edge.dstVertexID;		
		//fin>>edgeWeight;

		getline(fin,temp_str);
	}
	fin.close();
}


/** Unweighted and undirected graph reader
*	File Format
Number_Of_Vertices Number_Of_Edges
src	dst
src	dst
...
...
src	dst
*/
void UWUDReader::ReadGraph(string file_name)
{
	ifstream fin(file_name.c_str());
	if(!fin.is_open())
	{
		printf("reading file failed\n");
		exit(-1);
	}
	int vertex_count, edge_count, src_id, dst_id;
	fin>>vertex_count>>edge_count;
	graph.resize(vertex_count);
	
	string temp_str;
	Edge temp_edge;
	while(fin>>src_id)
	{
		fin>>dst_id;
		getline(fin,temp_str);
		temp_edge.srcVertexID = src_id;
		temp_edge.dstVertexID = dst_id;
		graph.AddEdge(temp_edge);
		temp_edge.srcVertexID = dst_id;
		temp_edge.dstVertexID = src_id;
		graph.AddEdge(temp_edge);
	}
	fin.close();
}


/** Unweighted and directed graph reader
*	File Format
Number_Of_Vertices Number_Of_Edges
src	dst
src	dst
...
...
src	dst
*/
void UWDReader::ReadGraph(string file_name)
{
	ifstream fin(file_name.c_str());
	int vertex_count, edge_count, src_id, dst_id;
	fin>>vertex_count>>edge_count;
	graph.resize(vertex_count);

	string temp_str;
	Edge temp_edge;
	while(fin>>src_id)
	{
		fin>>dst_id;
		getline(fin,temp_str);
		temp_edge.srcVertexID = src_id;
		temp_edge.dstVertexID = dst_id;
		graph.AddEdge(temp_edge);
	}
	fin.close();
}

/**
*
*
* @param file_name - the name of the graph file
* @param weighted - true if the graph is weighed
* @param directed - true if the graph is directed
* Algorithm description:
The input src_v_id are not continuous and may not be sorted.
Put all the tuples in to an array, sort the array so tuples with
the same src_v_id is grouped together. Build the L2P map. Fill
the array into the graph.

* @note	
*
*/

bool cmp(Edge& l, Edge& r)
{
	return l.srcVertexID < r.srcVertexID;
}


/**
* generic graph reader, read all kinds of graph, the input format is:
src_id dst_id [weight]
src_id dst_id [weight]
src_id dst_id [weight]
.
.
.
.
src_id dst_id [weight]




*

* @param	- degree_threshold the threshold of vertex degree, vertex with threshold lower than this will not
				be included in the graph
* @param	- weight_threshold, edge with weight less than this threshold will not be included in the graph(this will NOT affect the vertex degree!!!)
				weight_threshold < 0 indicates the graph is not weighted, weight_threshold == 0 means the graph is weighted but the threshold is 0
* @return	
* @note	
*
*/



/**
* list<Edge>t_list - list representation of the input file
* perform BFS on the input file, unreached vertices and associated
* edges will be removed from t_list
* the number of vertices should be known
*
*/

vector<int> two_hop_logical_id;//the logical ids of vertexes within 2-hop of CC Chang. 允许重复
int *vertex_weight;//the weight of each vertex, should be built before filtering
int max_weight;
HASH_SPACE::hash_map<int ,int> global_L2P;


void BFS_Mark(list<Edge> &t_list, int root_logical_id, int bfs_level)
{
	int *level;
	int vertex_num = 0;
	list<Edge>::iterator it;
	for(it = t_list.begin(); it != t_list.end(); it ++)
	{
		if(it->srcVertexID > vertex_num)
			vertex_num = it->srcVertexID;
		if(it->dstVertexID > vertex_num)
			vertex_num = it->dstVertexID;
	}
	vertex_num ++;


	CPUMalloc((void**)&level, sizeof(int)*vertex_num);
	for(int i = 0; i < vertex_num; i ++)
		level[i] = -1;
	level[root_logical_id] = 0;
	two_hop_logical_id.push_back(root_logical_id);

	bool cont = false;
	int super_step = 0;
	while(true)
	{
		cont = false;
		for(it = t_list.begin(); it != t_list.end(); it ++)
		{
			int src_id = it->srcVertexID;
			//printf("src_id = %d\n",src_id );
			if(level[src_id] == super_step)
			{
				int dst_id = it->dstVertexID;
				if(level[dst_id] == -1)
				{
					//printf("update dst_id = %d\n",dst_id);
					level[dst_id] = super_step + 1;
					two_hop_logical_id.push_back(dst_id);
					cont = true;
				}
			}
		}
		if(!cont)
			break;
		super_step ++;
		if(super_step == bfs_level)
			break;
	}
}



int BFS_FIltering(list<Edge> &t_list, HASH_SPACE::hash_map<int ,int> &L2P, int root_logical_id, int bfs_level)
{
	int *level;
	int vertex_num = 0;
	list<Edge>::iterator it;
	for(it = t_list.begin(); it != t_list.end(); it ++)
	{
		if(it->srcVertexID > vertex_num)
			vertex_num = it->srcVertexID;
		if(it->dstVertexID > vertex_num)
			vertex_num = it->dstVertexID;
	}
	vertex_num ++;


	CPUMalloc((void**)&level, sizeof(int)*vertex_num);
	for(int i = 0; i < vertex_num; i ++)
		level[i] = -1;
	level[root_logical_id] = 0;

	bool cont = false;
	int super_step = 0;
	while(true)
	{
		cont = false;
		for(it = t_list.begin(); it != t_list.end(); it ++)
		{
			int src_id = it->srcVertexID;
			//printf("src_id = %d\n",src_id );
			if(level[src_id] == super_step)
			{
				int dst_id = it->dstVertexID;
				if(level[dst_id] == -1)
				{
					//printf("update dst_id = %d\n",dst_id);
					level[dst_id] = super_step + 1;
					cont = true;
				}
			}
		}
		if(!cont)
			break;
		super_step ++;
		if(super_step == bfs_level)
			break;
	}


	int reached_vertex_num = 0;
	for(int i = 0; i < vertex_num; i ++)
	{
		if(level[i] >= 0)
			reached_vertex_num ++;
	}
	int reached_edge_num = 0;

	//rebuild the edge list and L2P
	global_L2P = L2P;
	int *new_vertex_weight;
	CPUMalloc((void**)&new_vertex_weight, sizeof(int)*L2P.size());
	memset(new_vertex_weight, 0, sizeof(int)*L2P.size());
	L2P.clear();
	list<Edge> new_t_list;
	int vertex_index = 0;
	for(it = t_list.begin(); it != t_list.end(); it ++)
	{
		if(level[it->srcVertexID] >= 0 && level[it->dstVertexID] >= 0)
		{

			Edge temp;
			temp.srcVertexID = it->srcVertexID;
			temp.dstVertexID = it->dstVertexID;
			temp.weight = it->weight;
			new_t_list.push_back(temp);
			reached_edge_num ++;
			if(L2P.find(it->srcVertexID) == L2P.end())
			{
				L2P[it->srcVertexID] = vertex_index ++;
				int my_weight = vertex_weight[global_L2P[it->srcVertexID]];
				new_vertex_weight[L2P[it->srcVertexID]] = my_weight;
			}
			if(L2P.find(it->dstVertexID) == L2P.end())
			{
				L2P[it->dstVertexID] = vertex_index ++;
				int my_weight = vertex_weight[global_L2P[it->dstVertexID]];
				new_vertex_weight[L2P[it->dstVertexID]] = my_weight;
			}
		}
	}

	t_list = new_t_list;
	free(vertex_weight);
	vertex_weight = new_vertex_weight;
	global_L2P.clear();
	free(level);

	/* check error */
	if(vertex_index != reached_vertex_num)
	{
		printf("error %d != %d\n",vertex_index, reached_vertex_num);
		exit(-1);
	}
	printf("reached_vertex_num = %d, reached_edge_num = %d\n",reached_vertex_num , reached_edge_num);

	return reached_vertex_num;
}




void GenericReader::ReadGraph(string file_name, bool directed, int degree_threashold, int weight_threshold, bool sorted,HASH_SPACE::hash_map<int ,int> &L2P)
{
	list<Edge>t_list;
	ifstream fin(file_name.c_str());


	Edge temp_tup;
	Edge temp_tup_re;
	int vertex_total_num = 0;
	int edge_rendered = 0;
	while(fin>>temp_tup.srcVertexID)
	{
		fin>>temp_tup.dstVertexID;


		/* find the total number of vertices */
		if(L2P.find(temp_tup.srcVertexID) == L2P.end())
		{
			L2P[temp_tup.srcVertexID] = vertex_total_num ++;
		}
		if(L2P.find(temp_tup.dstVertexID) == L2P.end())
		{
			L2P[temp_tup.dstVertexID] = vertex_total_num ++;
		}



		if(weight_threshold >= 0)
			fin>>temp_tup.weight;

		t_list.push_back(temp_tup);
		if(!directed)//must insert two edges for undirected graph
		{
			temp_tup_re.srcVertexID = temp_tup.dstVertexID;
			temp_tup_re.dstVertexID = temp_tup.srcVertexID;
			temp_tup_re.weight = temp_tup.weight;
			t_list.push_back(temp_tup_re);
		}
	}

	//printf("list built\n");

	//build the vertex weight (be in direct proportion to the number of publications)
	//weight是按physical id存的
	list<Edge>::iterator it;

#ifdef VIS
	CPUMalloc((void**)&vertex_weight, sizeof(int)*L2P.size());
	memset(vertex_weight, 0, sizeof(int)*L2P.size());
	for(it = t_list.begin(); it != t_list.end(); it ++)
	{
		//由于是双向边，只需要src或dst加上边的权重
		vertex_weight[L2P[it->srcVertexID]] += it->weight;
	}
	max_weight = -1;
	for(int i = 0; i < L2P.size(); i ++)
	{
		if(vertex_weight[i] > max_weight)
			max_weight = vertex_weight[i];
	}
	printf(">>>>>>>>%d\n",max_weight);
#endif

//#define BFS_FILTERING
#ifdef BFS_FILTERING
	vertex_total_num = BFS_FIltering(t_list, L2P, 728850, 1);
#endif

//#define BFS_MARK
#ifdef BFS_MARK
//	unsigned int bfs_timer;
//	cutCreateTimer(&bfs_timer);
//	cutStartTimer(bfs_timer);
	//jiawei
	BFS_Mark(t_list, 728850, 2);
	//philip yu
	BFS_Mark(t_list, 398094, 2);
	//Raghu Ramakrishnan
	BFS_Mark(t_list, 151491, 2);
	//Rakesh Agrawal
	BFS_Mark(t_list, 697432, 2);
	//Christos Faloutsos
	BFS_Mark(t_list, 639921, 2);
	//Jennifer Widom
	BFS_Mark(t_list, 467668, 2);

	//	cutStopTimer(bfs_timer);
//	float t = cutGetTimerValue(bfs_timer);
//	printf("CPU BFS time = %f\n",t);
#endif


	//printf("list built size = %d\n",t_list.size());
	//printf("test_num = %d\n", vertex_total_num);

	/*  rebuild L2P map if need to filter the graph*/
	
	if(degree_threashold > 0)
	{
		/* rebuild L2P, since some vertices will be filterd */
		HASH_SPACE::hash_map<int ,int> new_L2P;
		int *degrees;
		
		CPUMalloc((void**)&degrees,sizeof(int)*vertex_total_num);
		
		memset(degrees, 0, sizeof(int)*vertex_total_num);
		int num_vertex = 0;//the number of vertex after filtering

		/* calculate the degree */
		it = t_list.begin();
		while(it != t_list.end())
		{
			//在有向图中为出度，无向图中就是度了(DBLP是无向图，故而对于每一篇论文，不管是第几作者，都会使其degree加1)
			if(L2P[it->srcVertexID] >= vertex_total_num)
			{
				printf("map error\n");
				exit(-1);
			}
			//不是图中的degree，而是作者一共有几篇文章，故使用weight
			degrees[L2P[it->srcVertexID]] += it->weight;
			it ++;
		}


		it = t_list.begin();
		while(it != t_list.end())
		{
			if(degrees[L2P[it->srcVertexID]] >= degree_threashold)
			{
				if(new_L2P.find(it->srcVertexID) == new_L2P.end())
					new_L2P[it->srcVertexID] = num_vertex ++;
			}

			if(degrees[L2P[it->dstVertexID]] >= degree_threashold)
			{
				if(new_L2P.find(it->dstVertexID) == new_L2P.end())
					new_L2P[it->dstVertexID] = num_vertex ++;
			}
			it ++;
		}
		vertex_total_num = num_vertex;
		free(degrees);
		L2P = new_L2P;
	}




	/* build GraphIR object from the list */
	graph.resize(vertex_total_num);
	printf("graph resized\n");
	
	while (!t_list.empty())
	{
		it = t_list.begin();
		if(L2P.find(it->srcVertexID) != L2P.end() && L2P.find(it->dstVertexID) != L2P.end())//if not filtered
		{
			Edge temp_edge;
			temp_edge.srcVertexID = it->srcVertexID;
			temp_edge.dstVertexID = it->dstVertexID;
			temp_edge.weight = it->weight;

			if((weight_threshold > 0 && temp_edge.weight >= weight_threshold) || weight_threshold <= 0)
			{
			
				temp_edge.srcVertexID = L2P[temp_edge.srcVertexID];
				temp_edge.dstVertexID = L2P[temp_edge.dstVertexID];
				graph.AddEdge(temp_edge);
				if(temp_edge.weight > 12)
					edge_rendered ++;
			}
		}
		t_list.pop_front();
	}
	//printf("edge_rendered = %d\n",edge_rendered);

	printf("GraphIR built %d vertexes %d edges\n",graph.vertexNum,graph.totalEdgeCount);
	//exit(-1);

}

#endif
