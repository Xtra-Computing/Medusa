#include "PartitionManager.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include "../MultipleGPU/Gather.cu"

#include <fstream>
#include <algorithm>
#include <map>
#include <cutil_inline.h>

long long make_64bit(int high, int low)
{
	long long temp = high;
	temp = temp << 32;
	long long temp2 = low;
	temp = temp | temp2;
	return temp;
}


GraphPartition::GraphPartition()
{

	routing_list = new list<pair<int, int> > [MGLOBAL::num_gpu_to_use]; //one list will not be used
	// i.e. , for the ith gpu, routing_list[i] will be empty
	max_hop = MGLOBAL::max_hop;
	replica_num = (int*)malloc(sizeof(int)*max_hop);
	memset(replica_num, 0, sizeof(int)*max_hop);


	replica_edge_num = (int*)malloc(sizeof(int)*max_hop);
	memset(replica_edge_num, 0, sizeof(int)*max_hop);

	replica_index = (int*)malloc(sizeof(int)*(max_hop + 1));
	memset(replica_index, 0, sizeof(int)*(max_hop + 1));


	replica_edge_index = (int*)malloc(sizeof(int)*(max_hop + 1));
	memset(replica_edge_index, 0, sizeof(int)*(max_hop + 1));


	gather_table_size = 0;
	scatter_table_size = 0;

	gather_seg_index = (int*)malloc(sizeof(int)*(MGLOBAL::num_gpu_to_use + 1));
	memset(gather_seg_index, 0, sizeof(int)*(MGLOBAL::num_gpu_to_use + 1));
	scatter_seg_index = (int*)malloc(sizeof(int)*(MGLOBAL::num_gpu_to_use + 1));
	memset(scatter_seg_index, 0, sizeof(int)*(MGLOBAL::num_gpu_to_use + 1));
}




/**
Data structure used for generate partitions
*/
#define P_ORIGIN 0
#define P_CORE 1
#define P_REPLICA 2

class VertexRecord
{
public:
	VertexRecord()
	{
		vertex_type = -1;
		self_edge_num = 0;
		par_id = -1;
		logical_id = -1;

		original_logical_id = -1;
		original_partition = -1;
	}
	int vertex_type; // 0 origin of replica, 1 core part, 2 replica
	int self_edge_num; // number of edges tailed at partition par_id
	int par_id; // belongs to which partition (numbered from 1)
	int logical_id; //id read from GT-graph file

	//these two attributes are only needed when the VertexRecord is a replica type
	int original_logical_id; //logical ID of its original
	int original_partition; // original partition
};

bool VertexRecordCmp(VertexRecord l, VertexRecord r)
{
	if(l.par_id < r.par_id)
		return true;
	else if(l.par_id > r.par_id)
		return false;
	else if(l.vertex_type < r.vertex_type)
		return true;
	else if(l.vertex_type > r.vertex_type)
		return false;

	else if(l.self_edge_num < r.self_edge_num )
		return true;
	else
		return false;
}

/**
*
* read the partition file to RAM, prepare for reading the graph file
*
*/
GTGraph::GTGraph()
{

	logical_id_to_partition = NULL;
}



/*
multi-hop version pf get_partition
*/
void GTGraph::get_parition(GraphPartition *gp_array, char *graph_file_name, char *partition_file_name)
{
	ifstream gt_file(graph_file_name);
	ifstream partition_file(partition_file_name);
	if(!gt_file.good())
	{
		printf("open GT Graph File failed\n");
		exit(-1);
	}
	if((!partition_file.good()) && (MGLOBAL::num_gpu_to_use != 1))
	{
		printf("open Partition File failed\n");
		exit(-1);
	}



	//read the partition file, check if it's consistent with the number of GPUs
	int temp_par = -1;
	int v_num_from_partition_file = 0;
	int partition_num_from_file = 0;
	if(MGLOBAL::num_gpu_to_use != 1)
	{
		while(partition_file >> temp_par)
		{
			if(temp_par > partition_num_from_file)
				partition_num_from_file = temp_par;
			v_num_from_partition_file ++;
		}
	}
	partition_num_from_file ++;//partition ID from the file is numbered from 0
	if((MGLOBAL::num_gpu_to_use != partition_num_from_file) && (MGLOBAL::num_gpu_to_use != 1))
	{
		printf("Error: The number of GPUs does not match the number of partitions\n!");
		exit(-1);
	}

	int v_num, e_num;
	char line[1024];
	char first_ch;
	while(gt_file.get(first_ch))
	{
		if(first_ch == 'p')
		{
			string temp;
			gt_file>>temp>>v_num>>e_num;
			gt_file.getline(line, 1024);//eat the line break
			break;
		}
		gt_file.getline(line, 1024);//eat the line break
	}

	//check whether the partition file is consistent with the gt_graph file
	if(v_num_from_partition_file != v_num && MGLOBAL::num_gpu_to_use != 1)
	{
		printf("vertex numbers from partition file and gtgraph file are different\n");
		exit(-1);
	}


	logical_id_to_partition = (int*)malloc((v_num + 1)*sizeof(int));
	if(MGLOBAL::num_gpu_to_use != 1)
	{
		partition_file.clear();
		partition_file.seekg(0, ios::beg);
	}

	int read_index = 1;

	if(MGLOBAL::num_gpu_to_use > 1)
		while(partition_file >> temp_par)
			logical_id_to_partition[read_index ++] = temp_par;
	else
		memset(logical_id_to_partition, 0, sizeof(int)*(v_num + 1));






	//read the graph into main memory (only works for continuous logical IDs [1, 2, ..., N])
	//log: 2012-1-12 To support reading weight, p_graphs type change from vector<int> to vector<pair<int, float> >
	vector<pair<int, float> > * p_graphs = new vector<pair<int, float> > [v_num + 1];
	vector<pair<int, float> > * reverse_p_graphs = new vector<pair<int, float> > [v_num + 1];

	int src_id, dst_id;
	float edge_weight;

	while(gt_file.get(first_ch))
	{
		if(first_ch == 'a')
		{
			gt_file>>src_id>>dst_id>>edge_weight;
			//printf("%d %d\n",src_id,dst_id);
			p_graphs[src_id].push_back(pair<int, float>(dst_id, edge_weight));
			reverse_p_graphs[dst_id].push_back(pair<int, float>(src_id, edge_weight));
		}
	}


	//allocate data structures
	bool **bfs_mark = (bool**)malloc(sizeof(bool*)*MGLOBAL::num_gpu_to_use);
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		bfs_mark[i] = (bool*)malloc(sizeof(bool)*(v_num + 1));
		memset(bfs_mark[i], 0, sizeof(bool)*(v_num + 1));
	}
	list<pair<int, int> > *bfs_queue = new list<pair<int, int> >[MGLOBAL::num_gpu_to_use];
	//gp_array.resize(MGLOBAL::num_gpu_to_use);//gp_array[i] corresponding to partition i


	//enqueue the core and boundary
	for(int v_index = 1; v_index <= v_num; v_index ++)
	{
		bfs_queue[logical_id_to_partition[v_index]].push_back(pair<int, int>(v_index, 0));
		//build l2p, NOTE: the l2p should be built in such a way that the vertices are sorted by outgoing degree.
		//However, they are currently not sorted
		int temp = gp_array[logical_id_to_partition[v_index]].l2p.size();
		gp_array[logical_id_to_partition[v_index]].l2p[v_index] = temp;//!!!!!!!!!!!!!!!
	}

	//get the number of non-replica vertices and also start position of the first replica
	for(int gpu_index = 0; gpu_index < MGLOBAL::num_gpu_to_use; gpu_index ++)
		gp_array[gpu_index].replica_index[0] = gp_array[gpu_index].l2p.size();


	//make a record if a vertex is a non-0-hop replica of other partitions
	//the destination partition of this vertex is push_back onto its list
	// <partition id, level>
	list<pair<int, int> > *non_last = new list<pair<int, int> >[v_num + 1];



	for(int gpu_index = 0; gpu_index < MGLOBAL::num_gpu_to_use; gpu_index ++)
	{

		//dequeue and build routing_list
		//record on dequeue
		while(!bfs_queue[gpu_index].empty())
		{
			pair<int, int> queue_head = bfs_queue[gpu_index].front();
			bfs_queue[gpu_index].pop_front();
			vector<pair<int, float> >::iterator it = reverse_p_graphs[queue_head.first].begin();
			while(it != reverse_p_graphs[queue_head.first].end())
			{
				if(bfs_mark[gpu_index][it->first] == true || logical_id_to_partition[it->first] == gpu_index)
				{
					it ++;
					continue;
				}

				bfs_mark[gpu_index][it->first] = true;
				gp_array[gpu_index].replica_num[queue_head.second] ++;



				gp_array[logical_id_to_partition[it->first]].routing_list[gpu_index].push_back(pair<int, int>(it->first, queue_head.second));

				if(queue_head.second != MGLOBAL::max_hop - 1)
					non_last[it->first].push_back(pair<int,int>(gpu_index, queue_head.second));


				if(queue_head.second + 1< MGLOBAL::max_hop)
					bfs_queue[gpu_index].push_back(pair<int, int>(it->first, queue_head.second + 1));

				it ++;
			}
		}

	}


	//print routing table for debugging
	/*
	for(int gpu_index = 0; gpu_index < MGLOBAL::num_gpu_to_use; gpu_index ++)
	{
	for(int gpu_index2 = 0; gpu_index2 < MGLOBAL::num_gpu_to_use; gpu_index2 ++)
	printf("GPU %d table %d: %d\n",gpu_index, gpu_index2, gp_array[gpu_index].routing_list[gpu_index2].size());
	}
	*/





	//combine the routing_list to make gather and scatter table for all the partitions (GPUs)
	for(int gpu_index = 0; gpu_index < MGLOBAL::num_gpu_to_use; gpu_index ++)
	{
		//calculate gather table size and gather index
		int gather_table_size = 0;
		for(int cal_index = 0; cal_index < MGLOBAL::num_gpu_to_use; cal_index ++)
		{
			gather_table_size += gp_array[gpu_index].routing_list[cal_index].size();
			gp_array[gpu_index].gather_seg_index[cal_index + 1] = gp_array[gpu_index].gather_seg_index[cal_index] + gp_array[gpu_index].routing_list[cal_index].size();
			//printf("GPU %d gather_seg_index[%d] = %d\n", gpu_index, cal_index + 1,gp_array[gpu_index].gather_seg_index[cal_index + 1]);
		}
		gp_array[gpu_index].gather_table = (int *)malloc(sizeof(int)*gather_table_size);
		gp_array[gpu_index].gather_table_size = gather_table_size;

		//build gather table
		list<pair<int, int> >::iterator it;
		int g_t_index = 0;
		for(int g_i = 0; g_i < MGLOBAL::num_gpu_to_use; g_i ++)
		{
			it = gp_array[gpu_index].routing_list[g_i].begin();
			while(it != gp_array[gpu_index].routing_list[g_i].end())
			{
				gp_array[gpu_index].gather_table[g_t_index ++] = gp_array[gpu_index].l2p[it->first];
				it ++;
			}
		}




		//calculate scatter table size and scatter index
		int scatter_table_size = 0;
		for(int cal_index = 0; cal_index < MGLOBAL::num_gpu_to_use; cal_index ++)
		{
			scatter_table_size += gp_array[cal_index].routing_list[gpu_index].size();
			gp_array[gpu_index].scatter_seg_index[cal_index + 1] = scatter_table_size;
			//printf("GPU %d scatter_seg_index[%d] = %d\n",gpu_index, cal_index + 1, gp_array[gpu_index].scatter_seg_index[cal_index + 1]);
		}
		gp_array[gpu_index].scatter_table = (int *)malloc(sizeof(int)*scatter_table_size);
		gp_array[gpu_index].scatter_table_size = scatter_table_size;

		list<int> *temp_collect = new list<int> [MGLOBAL::max_hop];
		for(int g_i = 0; g_i < MGLOBAL::num_gpu_to_use; g_i ++)
		{

			it = gp_array[g_i].routing_list[gpu_index].begin();
			while(it != gp_array[g_i].routing_list[gpu_index].end())
			{
				temp_collect[(*it).second].push_back((*it).first);	
				it ++;
			}
		}
		//生成physical id
		for(int ti = 0; ti < MGLOBAL::max_hop; ti ++)
		{

			//pure debug
			if(temp_collect[ti].size() != gp_array[gpu_index].replica_num[ti])
			{
				printf("replica number error %d != %d\n",temp_collect[ti].size(), gp_array[gpu_index].replica_num[ti]);
			}

			list<int>::iterator lt;
			lt = temp_collect[ti].begin();
			while(lt != temp_collect[ti].end())
			{
				int temp = gp_array[gpu_index].l2p.size();
				gp_array[gpu_index].l2p[*lt] = temp;//!!!!!!!!!!!!!!!
				lt ++;
			}
		}


		//build scatter table
		int s_t_index = 0;
		for(int g_i = 0; g_i < MGLOBAL::num_gpu_to_use; g_i ++)
		{

			it = gp_array[g_i].routing_list[gpu_index].begin();
			while(it != gp_array[g_i].routing_list[gpu_index].end())
			{
				gp_array[gpu_index].scatter_table[s_t_index ++] = gp_array[gpu_index].l2p[it->first];
				it ++;
			}
		}

	}

	//build replica index
	for(int gpu_id = 0; gpu_id < MGLOBAL::num_gpu_to_use; gpu_id ++)
	{
		for(int i = 1; i <= MGLOBAL::max_hop; i ++)
			gp_array[gpu_id].replica_index[i] = gp_array[gpu_id].replica_num[i - 1] + gp_array[gpu_id].replica_index[i - 1];
	}

	printf("start to add edge\n");

	//build the GraphIR for each partition
	//construct the partition tree based on L2P
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		gp_array[i].graph.resize(gp_array[i].replica_index[MGLOBAL::max_hop]);

	int *added_edge_num = (int*) malloc(sizeof(int)*MGLOBAL::num_gpu_to_use);
	memset(added_edge_num, 0, sizeof(int)*MGLOBAL::num_gpu_to_use);
	for(int v_index = 1; v_index <= v_num; v_index ++)
	{
		vector<pair<int, float> >::iterator int_it;
		int head_partition = logical_id_to_partition[v_index];
		int_it = p_graphs[v_index].begin();
		while(int_it != p_graphs[v_index].end())
		{

			int tail_partition = logical_id_to_partition[int_it->first];

		
			if(head_partition == tail_partition)
			{
				int src_p_id = gp_array[tail_partition].l2p[v_index];
				int dst_p_id = gp_array[tail_partition].l2p[int_it->first];
				Edge temp_edge;
				temp_edge.srcVertexID = src_p_id;
				temp_edge.dstVertexID = dst_p_id;
				temp_edge.weight = int_it->second;
				gp_array[tail_partition].graph.AddEdge(temp_edge);
				added_edge_num[tail_partition] ++;
				gp_array[tail_partition].replica_edge_index[0] ++;//非replica的边作为replica edge的offset


			}
			else
			{
				int src_p_id = gp_array[tail_partition].l2p[v_index];
				int dst_p_id = gp_array[tail_partition].l2p[int_it->first];
				Edge temp_edge;
				temp_edge.srcVertexID = src_p_id;
				temp_edge.dstVertexID = dst_p_id;
				temp_edge.weight = int_it->second;
				gp_array[tail_partition].graph.AddEdge(temp_edge);
				added_edge_num[tail_partition] ++;
				gp_array[tail_partition].replica_edge_num[0] ++;

			}



			list<pair<int, int> >::iterator parit;
			parit = non_last[int_it->first].begin();
			while(parit != non_last[int_it->first].end())
			{

				int tail_partition = (*parit).first;
				int src_p_id = gp_array[tail_partition].l2p[v_index];
				int dst_p_id = gp_array[tail_partition].l2p[int_it->first];
				Edge temp_edge;
				temp_edge.srcVertexID = src_p_id;
				temp_edge.dstVertexID = dst_p_id;
				temp_edge.weight = int_it->second;
				gp_array[tail_partition].graph.AddEdge(temp_edge);


				added_edge_num[tail_partition] ++;

				int src_level_count = 0;
				while(src_p_id > gp_array[tail_partition].replica_index[src_level_count])
					src_level_count ++;
				if(src_level_count == 0)
					gp_array[tail_partition].replica_edge_index[0] ++;
				else
					gp_array[tail_partition].replica_edge_num[src_level_count - 1] ++;



				parit ++;
			}


			int_it ++;
		}

	}



	//build replica edge index
	for(int gpu_id = 0; gpu_id < MGLOBAL::num_gpu_to_use; gpu_id ++)
	{
		printf("gp_array[%d].replica_edge_index[0] = %d\n", gpu_id, gp_array[gpu_id].replica_edge_index[0]);
		for(int i = 1; i <= MGLOBAL::max_hop; i ++)
		{
			gp_array[gpu_id].replica_edge_index[i] = gp_array[gpu_id].replica_edge_num[i -1] + gp_array[gpu_id].replica_edge_index[i - 1];	
		}

	}





	//for debug, print total number of stored edges
	int total_stored_edges = 0;
	for(int pid = 0; pid < MGLOBAL::num_gpu_to_use; pid ++)
		total_stored_edges += added_edge_num[pid];
	printf("total number of stored edges %d\n", total_stored_edges);
	//end of for debug



	//for debug, print replica num and replica edge num

	for(int gpu_id = 0; gpu_id < MGLOBAL::num_gpu_to_use; gpu_id ++)
	{
		if(gp_array[gpu_id].l2p.size() != gp_array[gpu_id].replica_index[MGLOBAL::max_hop])
		{
			printf("gpu_id %d %d != %d\n", gpu_id, gp_array[gpu_id].l2p.size(), gp_array[gpu_id].replica_index[MGLOBAL::max_hop]);
			exit(-1);
		}


		//printf("GPU %d with %d non-replica vertex %d non-replica edge, added %d edges:\n", gpu_id, gp_array[gpu_id].replica_index[0],gp_array[gpu_id].replica_edge_index[0], added_edge_num[gpu_id]);
		for(int i = 0; i < MGLOBAL::max_hop; i ++)	
		{
			printf("level %d: %d vertex %d edge\n",i , gp_array[gpu_id].replica_num[i], gp_array[gpu_id].replica_edge_num[i]);
		}
	}

	


	delete [] p_graphs;
	
	gt_file.close();
	if(MGLOBAL::num_gpu_to_use != 1)
		partition_file.close();
}
