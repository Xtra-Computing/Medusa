/*

*/
#ifndef PARTITION_MANAGER_H
#define PARTITION_MANAGER_H

#include "../MedusaRT/GraphConverter.h"
#include <vector>
#include <string>
#include <utility>
#include <list>
#ifdef _WIN32
#include <hash_map>
#define HASH_SPACE stdext
#endif

#ifdef __linux__
#include <ext/hash_map>
#define HASH_SPACE __gnu_cxx
#endif

using namespace std;

class GraphPartition
{
	public:
		GraphPartition();
		GraphIR graph;


		int *gather_table; //the size of this array is the same as the number of replica in this partition
		int gather_table_size;
		int *scatter_table;
		int scatter_table_size;
/*
		why not do the gather/scatter in GPU?
		Since the communication pattern is all-to-all, N*(N - 1) pairs of gather and scatter need to be performed.
		Also, (N - 1) gather and scatter table need to be stored on each GPU. This is obviously to costly.
*/


		//for GPU-to-GPU routing
		//<logical_id, replica_level>
		list<pair<int, int> > *routing_list; //routing list to other N - 1 GPUs <gather from this gpu, scatter to the target gpu>
		HASH_SPACE::hash_map<int, int> l2p; //Logical ID from file to physical ID for each partition
											//The hash space is arranged as [ non-replica | replica]
		int *gather_seg_index; //gather_seg_index[i] start position of origins to GPU i (size == GPU_num + 1)
		int *scatter_seg_index; //scatter_seg_index[i] start position of origins from GPU i (size == GPU_num + 1)


		//for multi-hop processing
		int *replica_num;//record the number of replica for each hop
		int *replica_index;//replica_index[0] == the number of non-replica vertices, also start position of the first level replica
		int *replica_edge_num;// record the number of edges of replicas in each hop
		int *replica_edge_index;//replica_edge_index[0] == the number of non-replica vertices' edges, also start position of the edges of first level replica

		int max_hop; //the number of hops for replication

};


class GTGraph
{
public:
	GTGraph();
	void get_parition(GraphPartition *gp_array, char *graph_file_name, char *partition_file_name);

private:
	int *logical_id_to_partition;
};

/**
*	First copy the updated originals to main memory, shuffle,a and copy to replicas
*
* @param  syn - this function use asynchronous memcpy calls. If syn is true, a synchronization function is called
				at the end of this function.
* @return	
* @note	
*
*/
void *UpdateReplica(void *syn);


#endif