#include <vector>
#ifdef _WIN32
#include <hash_map>
#endif
#ifdef __linux__
#include <ext/hash_map>
#endif
#include "../MedusaRT/GraphGenerator.h"
#include "../MedusaRT/GraphConverter.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include "../MedusaRT/SystemLibCPU.h"
#include "../MultipleGPU/MultiPublicAPI.h"
#include "../MedusaRT/GraphGenerator.h"
#include "../MedusaRT/GraphReader.h"
#include "Configuration.h"
#include "CPUFunctorHoster.h"
#include "DeviceDataStructure.h"
#include "../MultipleGPU/PartitionManager.h"
#include "../MultipleGPU/MultiUtilities.h"
#include <fstream>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline_runtime.h>


#include "../Tools/ReplicaNumberAnalysis.h"

#include <float.h>
/**
* SSSP algorithm on CPU, used to compare results with the GPU algorithm
* Implementation: Dijkstra's + linked list
* @param - graph: the input graph
* @param - dis: distance of each vertex, allocated inside the function
* @param - root_id: the root vertex id
* @return	
* @note	
*
*/


struct int_pair
{
	int x;
	float y;
	int_pair()
	{
		x = 0;
		y = 0;
	}
	int_pair(int _x, float _y)
	{
		x = _x;
		y = _y;
	}

};

struct int_pair_node
{
	int_pair pair;
	int_pair_node *next;
};

struct int_pair_list
{
	int_pair_node *head;
	int_pair_node *end;
	int size;
	int_pair_list()
	{
		head = NULL;
		end = NULL;
		size = 0;
	}
	void push_back(int_pair _pair)
	{
		int_pair_node *temp = (int_pair_node *)malloc(sizeof(int_pair_node));
		temp->pair = _pair;
		temp->next = NULL;
		if(size)
			end->next = temp;
		else
		{
			head = temp;
		}
		end = temp;
		size ++;
	}
	int_pair pop_front()
	{
		if(size <= 0)
		{
			printf("pop error\n");
			exit(-1);
		}
		size --;
		int_pair temp = head->pair;
		head = head ->next;
		return temp;

	}
	bool empty()
	{
		if(size > 0)
			return false;
		else
			return true;
	}
	//remove the node with minimum distance
	//if all distance equals to MEDUSA_MAX, return NULL
	//return the next node of the deleted node
	int_pair_node* remove_minimum(float *distance)
	{
		//printf("size = %d\n",size);
		//the list is empty
		if(head == NULL)
		{
			return head;
		}
		int_pair_node* precedent_of_min = NULL;
		int_pair_node* precdent_of_current = head;
		int_pair_node* current_ptr = head->next;

		//the list has only one element
		if(current_ptr == NULL)
		{
			if(distance[head->pair.x] != MEDUSA_MAX)
			{
				size --;
				int_pair_node *temp = head;
				head = end = NULL;
				return temp;
			}
			else
				return NULL;
		}

		float min = distance[head->pair.x];
		while(current_ptr != NULL)
		{
			if(distance[current_ptr->pair.x] < min)
			{
				precedent_of_min = precdent_of_current;
				min = distance[current_ptr->pair.x];
			}
			precdent_of_current = current_ptr;
			current_ptr = current_ptr->next;
		}
		//if head is the min
		if(precedent_of_min == NULL)
		{
			size --;
			int_pair_node *temp = head;
			head = head->next;
			return temp;
		}
		if(distance[precedent_of_min->next->pair.x] != MEDUSA_MAX)
		{
			size --;
			int_pair_node *temp = precedent_of_min->next;
			precedent_of_min->next = precedent_of_min->next->next;
			if(precedent_of_min->next == NULL)
				end = precedent_of_min;
			return temp;
		}
		else
			return NULL;
	}

};

void CPUSSSP(GraphIR &graph, float **dis, int root_id)
{
	*dis = (float*)malloc(sizeof(float)*graph.vertexNum);
	for(int i = 0; i < graph.vertexNum; i ++)
		(*dis)[i] = MEDUSA_MAX;
	(*dis)[root_id] = 0;
	int_pair temp;
	int_pair_list v_list;
	for(int i = 0; i < graph.vertexNum; i ++)
	{
		temp.x = i;
		v_list.push_back(temp);
	}
	int_pair_node *min_node = v_list.remove_minimum(*dis);


	while(min_node != NULL)
	{
		if((*dis)[min_node->pair.x] == MEDUSA_MAX)
			break;
		//printf("min dis=%f\n",(*dis)[min_node->pair.x]);
		//update min_node's neighbors
		int v_id = min_node->pair.x;
		float curr_dis = (*dis)[v_id];
		EdgeNode *tempEdgeNode = graph.vertexArray[v_id].firstEdge;
		while(tempEdgeNode != NULL)
		{
			int n_id = tempEdgeNode->edge.dstVertexID;
			if(curr_dis + tempEdgeNode->edge.weight < (*dis)[n_id])
				(*dis)[n_id] = curr_dis + tempEdgeNode->edge.weight;
			tempEdgeNode = tempEdgeNode->nextEdge;
		}
		//get the next min
		min_node = v_list.remove_minimum(*dis);

	}
}



#define MULTIPLE_RUM
#define RUN_TIMES 1
#define CPU_COUNTER_PART
int main(int argc, char **argv)
{

	if(argc != 5)
	{
		printf("Usage: Medusa GPU_Num Max_hop Gt_File_Name Partition_File_Name\n");
		exit(-1);
	}


	//global configuration
	InitConfig(argv[1], argv[2], EDGE_MESSAGE, false);
	MGLOBAL::combiner_datatype = CUDPP_FLOAT;
	MGLOBAL::combiner_operator = CUDPP_MIN;

	//print multiple hop statistics
	//AnalyzeReplica(argv[3], argv[4]);


	float t;//timer value
	unsigned int timer = 0;
	cutCreateTimer(&timer);


	GTGraph gt_graph;
	//vector<GraphPartition> gp_array;
	GraphPartition *gp_array = new GraphPartition[MGLOBAL::num_gpu_to_use];
	gt_graph.get_parition(gp_array, argv[3], argv[4]);

	//set up data structures for all GPUs
	/* <algorithm specific initialization>  */
	srand(528);
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);	


		for(int vertex_index = 0; vertex_index < gp_array[i].graph.vertexNum; vertex_index ++)
		{
			EdgeNode *tempEdgeNode = gp_array[i].graph.vertexArray[vertex_index].firstEdge;
			while(tempEdgeNode != NULL)
			{
				tempEdgeNode->edge.weight = rand32()%1000;
				tempEdgeNode = tempEdgeNode->nextEdge;
			}
		}
	}
	



	InitHostDS(gp_array);
	//CheckOutgoingEdge(gp_array);
	printf("InitHostDS Done\n");



	//find the logical ID of the vertex with largest out-going degree
	ifstream gt_file(argv[3]);

	int total_v_num = 0, total_e_num = 0;
	char first_ch;
	char line[1024];
	while(gt_file.get(first_ch))
	{
		if(first_ch == 'p')
		{
			string temp;
			gt_file>>temp>>total_v_num;
			gt_file.getline(line, 1024);//eat the line break
			break;
		}
		gt_file.getline(line, 1024);//eat the line break
	}
	int src_id, dst_id;
	float edge_weight;
	int *v_edge_num = (int *)malloc(sizeof(int)*(total_v_num + 1));
	memset(v_edge_num, 0, sizeof(int)*(total_v_num + 1));
	while(gt_file.get(first_ch))
	{
		if(first_ch == 'a')
		{
			gt_file>>src_id>>dst_id>>edge_weight;
			v_edge_num[src_id] ++;
		}
	}
	int largest_vertex = 1;
	for(int i = 2; i <= total_v_num; i ++)
		if(v_edge_num[i] > v_edge_num[largest_vertex])
			largest_vertex = i;

	//end of find the logical ID
	int large_vertex_pid;

	/* <algorithm specific initialization>  */
#ifdef MULTIPLE_RUM
	for(int run_index = 0; run_index < RUN_TIMES; run_index ++)
	{

#endif
		//initialize distance
		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			medusaSetDevice(i);

			//MGLOBAL::gpu_def[i].edgeArray.buildAA(gp_array[i].graph);

			for(int vertex_index = 0; vertex_index < MGLOBAL::gpu_def[i].vertexArray.size; vertex_index++)
			{
				MGLOBAL::gpu_def[i].vertexArray.distance[vertex_index] = MEDUSA_MAX;
				MGLOBAL::gpu_def[i].vertexArray.updated[vertex_index] = false;
			}
		}

		//find the physical id of largest_vertex and set root

		for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
		{
			large_vertex_pid  = -1;
			HASH_SPACE::hash_map<int, int>::iterator it;
			it = gp_array[i].l2p.begin();
			while(it != gp_array[i].l2p.end())
			{
				if(it->first == largest_vertex)
				{
					large_vertex_pid = it->second;
					break;
				}
				it ++;
			}
			if(large_vertex_pid != -1)
			{
				MGLOBAL::gpu_def[i].vertexArray.distance[large_vertex_pid] = 0.0;
				MGLOBAL::gpu_def[i].vertexArray.updated[large_vertex_pid] = true;
				printf("Medusa Set SSSP Root: set %d(degree:%d) in partition %d\n", large_vertex_pid, v_edge_num[largest_vertex], i);

			}
		}
		gt_file.close();

		/* </algorithm specific initialization>  */	


		//call this function when all vertex and edge data are ready
#ifdef MULTIPLE_RUM
		if(run_index == 0)
			InitDeviceDS();
		else
			ResetGraphData();
#else
		InitDeviceDS();
#endif


		bool exe_temp = true;
		cutResetTimer(timer);
		cutStartTimer(timer);
		while(exe_temp)
		{
			//reset the global GPU side toExecute variable
			ResetToExecute();

			Medusa_Exec();

			exe_temp = RetriveToExecute();
			//inc super_step
			(MGLOBAL::super_step) ++;
		}
		cutStopTimer(timer);


#ifdef MULTIPLE_RUM
	}
#endif	

	t = cutGetAverageTimerValue(timer);
	printf("Medusa step %d\n", MGLOBAL::super_step);
	printf("Medusa SSSP %.3f ms\n",t); 



	HASH_SPACE::hash_map<int, int>::iterator it;
#ifdef CPU_COUNTER_PART
	//<Test CPU Counter Part>
	if(MGLOBAL::num_gpu_to_use != 1)
	{
		printf("Test CPU counter part, only accept one partition\n");
		exit(-1);
	}


	//find the physical id of largest_vertex and set root
	large_vertex_pid  = -1;
	it = gp_array[0].l2p.begin();
	while(it != gp_array[0].l2p.end())
	{
		if(it->first == largest_vertex)
		{
			large_vertex_pid = it->second;
			break;
		}
		it ++;
	}

	Init_CPU_Medusa(MGLOBAL::gpu_def[0].d_messageArray.size);


	cutResetTimer(timer);
	for(int test_count = 0; test_count < RUN_TIMES; test_count ++)
	{
		//init distances and set the root
		//initialize distance
		//MGLOBAL::gpu_def[0].edgeArray.buildAA(gp_array[0].graph);

		for(int vertex_index = 0; vertex_index < MGLOBAL::gpu_def[0].vertexArray.size; vertex_index++)
		{
			MGLOBAL::gpu_def[0].vertexArray.distance[vertex_index] = MEDUSA_MAX;
			MGLOBAL::gpu_def[0].vertexArray.updated[vertex_index] = false;
		}


		if(large_vertex_pid != -1)
		{
			MGLOBAL::gpu_def[0].vertexArray.distance[large_vertex_pid] = 0.0;
			MGLOBAL::gpu_def[0].vertexArray.updated[large_vertex_pid] = true;
			printf("CPU SSSP Root: set %d in partition %d\n", large_vertex_pid, 0);
		}

		cutStartTimer(timer);
		Medusa_Exec_CPU(MGLOBAL::gpu_def[0].vertexArray, MGLOBAL::gpu_def[0].edgeArray);
		cutStopTimer(timer);
	}
	t = cutGetTimerValue(timer)/RUN_TIMES;
	printf("Medusa CPU SSSP %f ms\n", t);
	//</Test CPU Counter Part>

#endif


	if(MGLOBAL::num_gpu_to_use != 1)
	{
		printf("Test IIIT implementation, only accept one partition\n");
		exit(-1);
	}

	t = 0.0;
	double total_distance = 0;
	//save the previous results for comparison
	float *cpu_distance = (float *)malloc(sizeof(float)*MGLOBAL::gpu_def[0].vertexArray.size);
	memcpy(cpu_distance, MGLOBAL::gpu_def[0].vertexArray.distance, sizeof(float)*MGLOBAL::gpu_def[0].vertexArray.size);
	MGLOBAL::gpu_def[0].d_vertexArray.Dump(MGLOBAL::gpu_def[0].vertexArray);
	for(int i = 0; i < MGLOBAL::gpu_def[0].d_vertexArray.size; i ++)
	{
		if(MGLOBAL::gpu_def[0].vertexArray.distance[i]!=MEDUSA_MAX)
		total_distance += double(MGLOBAL::gpu_def[0].vertexArray.distance[i]);

		if(abs(cpu_distance[i] - MGLOBAL::gpu_def[0].vertexArray.distance[i]) > 0.001)
		{
			printf("!!![%d]%f!=%f\n",i,cpu_distance[i], MGLOBAL::gpu_def[0].vertexArray.distance[i]);
			printf("Medusa CPU/GPU test error!\n");
			break;
		}
		//printf("[%d] %f ",i,cpu_distance[i]);
	}
	printf("total distance = %f\n", total_distance);

	return 0;
}
