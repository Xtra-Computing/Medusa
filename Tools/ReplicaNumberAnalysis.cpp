/*
	Analyze the number of replicas as the replica hop number increases




*/
#include "../Tools/ReplicaNumberAnalysis.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <list>
#include <string>
#include <cstring>
using namespace std;




void AnalyzeReplica(char *graph_file_name, char *partition_file_name)
{
	ifstream gt_file(graph_file_name);
	ifstream partition_file(partition_file_name);
	if(!gt_file.good())
	{
		printf("open GT Graph File failed\n");
		exit(-1);
	}
	if(!partition_file.good())
	{
		printf("open Partition File failed\n");
		exit(-1);
	}


	int temp_par = -1;
	int v_num_from_partition_file = 0;
	int partition_num_from_file = 0;
	while(partition_file >> temp_par)
	{
		if(temp_par > partition_num_from_file)
			partition_num_from_file = temp_par;
		v_num_from_partition_file ++;
	}
	partition_num_from_file ++;//partition ID from the file is numbered from 0
	if((MGLOBAL::num_gpu_to_use != partition_num_from_file) && (MGLOBAL::num_gpu_to_use != 1))
	{
		printf("Error: The number of GPUs does not match the number of partitions GPU %d Partition %d\n!", MGLOBAL::num_gpu_to_use, partition_num_from_file);
		exit(-1);
	}


	int *logical_id_to_partition = (int*)malloc((v_num_from_partition_file + 1)*sizeof(int));
	int *p_v_num = (int *)malloc(sizeof(int)*MGLOBAL::num_gpu_to_use);
	memset(p_v_num, 0, sizeof(int)*MGLOBAL::num_gpu_to_use);

	partition_file.clear();
	partition_file.seekg(0, ios::beg);
	int read_index = 1;

	if(MGLOBAL::num_gpu_to_use > 1)
		while(partition_file >> temp_par)
		{
			logical_id_to_partition[read_index ++] = temp_par;
			p_v_num[temp_par] ++;
		}
	else
		{
			memset(logical_id_to_partition, 0, sizeof(int)*(v_num_from_partition_file + 1));
			p_v_num[0] = v_num_from_partition_file;
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
	if(v_num_from_partition_file != v_num)
	{
		printf("vertex numbers from partition file and gt-graph file are different\n");
		exit(-1);
	}



	//read the graph into main memory (only works for continuous logical IDs [1, 2, ..., N])
	list<int> * p_graphs = new list<int> [v_num + 1];
	list<int> * reverse_p_graphs = new list<int> [v_num + 1];

	int src_id, dst_id;
	float edge_weight;

	while(gt_file.get(first_ch))
	{
		if(first_ch == 'a')
		{
			gt_file>>src_id>>dst_id>>edge_weight;
			p_graphs[src_id].push_back(dst_id);
			reverse_p_graphs[dst_id].push_back(src_id);
		}
	}


	//allocate data structures
	bool **bfs_mark = (bool**)malloc(sizeof(bool*)*MGLOBAL::num_gpu_to_use);
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		bfs_mark[i] = (bool*)malloc(sizeof(bool)*(v_num + 1));
		memset(bfs_mark[i], 0, sizeof(bool)*(v_num + 1));
	}

	int **replica_num = (int**)malloc(sizeof(int*)*MGLOBAL::num_gpu_to_use);


	int **replica_edge_num = (int**)malloc(sizeof(int*)*MGLOBAL::num_gpu_to_use);

	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		replica_num[i] = (int*)malloc(sizeof(int)*MGLOBAL::max_hop);
		memset(replica_num[i], 0, sizeof(int)*MGLOBAL::max_hop);

		replica_edge_num[i] = (int*)malloc(sizeof(int)*MGLOBAL::max_hop);
		memset(replica_edge_num[i], 0, sizeof(int)*MGLOBAL::max_hop);

	}

	list<pair<int, int> > *bfs_queue = new list<pair<int, int> >[MGLOBAL::num_gpu_to_use];
	//gp_array.resize(MGLOBAL::num_gpu_to_use);//gp_array[i] corresponding to partition i
	




	//enqueue the core and boundary
	for(int v_index = 1; v_index <= v_num; v_index ++)
	{
		bfs_queue[logical_id_to_partition[v_index]].push_back(pair<int, int>(v_index, 0));
	}

	//make a record if a vertex is a non-0-hop replica of other partitions
	//the destination partition of this vertex is push_back onto its list
	// <dst_partition, replica level>
	list<pair<int, int> > *non_last = new list<pair<int, int> >[v_num + 1];



	for(int gpu_index = 0; gpu_index < MGLOBAL::num_gpu_to_use; gpu_index ++)
	{

		//dequeue and build routing_list
		//record on dequeue
		while(!bfs_queue[gpu_index].empty())
		{
			pair<int, int> queue_head = bfs_queue[gpu_index].front();
			bfs_queue[gpu_index].pop_front();
			list<int>::iterator it = reverse_p_graphs[queue_head.first].begin();
			while(it != reverse_p_graphs[queue_head.first].end())
			{
				if(bfs_mark[gpu_index][*it] == true || logical_id_to_partition[*it] == gpu_index)
				{
					it ++;
					continue;
				}

				bfs_mark[gpu_index][*it] = true;
				//printf("Partition %d level %d reached %d through (%d->%d)\n", gpu_index, queue_head.second, *it, *it, queue_head.first);			

				replica_num[gpu_index][queue_head.second] ++;

				if(queue_head.second != MGLOBAL::max_hop - 1)
					non_last[*it].push_back(pair<int, int>(gpu_index, queue_head.second));

				if(queue_head.second + 1< MGLOBAL::max_hop)
					bfs_queue[gpu_index].push_back(pair<int, int>(*it, queue_head.second + 1));

				it ++;
			}
		}

	}



	for(int v_index = 1; v_index <= v_num; v_index ++)
	{
		list<int>::iterator int_it;
		int head_partition = logical_id_to_partition[v_index];
		int_it = p_graphs[v_index].begin();
		while(int_it != p_graphs[v_index].end())
		{

			int tail_partition = logical_id_to_partition[*int_it];


			if(head_partition == tail_partition)
			{

				//



			}
			else
			{

				replica_edge_num[tail_partition][0] ++;
			}
			list<pair<int, int> >::iterator parit;
			parit = non_last[*int_it].begin();
			while(parit != non_last[*int_it].end())
			{
			
				int tail_partition = (*parit).first;
				replica_edge_num[(*parit).first][(*parit).second + 1] ++;



				parit ++;

			}




			int_it ++;
		}
	}


	//print statistics
	for(int p_index = 0; p_index < MGLOBAL::num_gpu_to_use; p_index ++)
	{
		printf("Partition %d has %d vertices\n", p_index, p_v_num[p_index]);

		for(int hop_index = 0; hop_index < MGLOBAL::max_hop; hop_index ++)
		{
			printf("	%d hop has %d replicas %d edges\n", hop_index, replica_num[p_index][hop_index], replica_edge_num[p_index][hop_index]);
		}
	}

}
