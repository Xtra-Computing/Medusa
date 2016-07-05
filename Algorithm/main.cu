#include <vector>
#ifdef _WIN32
#include <hash_map>
#endif
#ifdef __linux__
#include <ext/hash_map>
#endif
#include "../MedusaRT/GraphGenerator.h"
#include "../MedusaRT/GraphConverter.h"
#include "../MedusaRT/GraphReader.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include "../MedusaRT/SystemLibCPU.h"
#include "../Algorithm/DeviceDataStructure.h"
#include "../MultipleGPU/MultiUtilities.h"
#include "../MultipleGPU/MultiPublicAPI.h"
#include "Configuration.h"
#include "CPUFunctorHoster.h"
#include "../MultipleGPU/PartitionManager.h"
#include "../Tools/ReplicaNumberAnalysis.h"

#include "../Compatibility/Compatability.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <cuda_runtime.h>
#include <fstream>



struct int_pair
{
	int x,y;
	int_pair(int _x, int _y)
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

};

void CPUBFS(GraphIR &graph, int **level, int root)
{
	CPUMalloc((void**)level, sizeof(int)*graph.vertexNum);
	for(int i = 0; i < graph.vertexNum; i ++)
		(*level)[i] = -1;
	(*level)[root] = 0;

	int vertexCount = 0;
	int currentLevel = 0;

	int_pair_list v_list;
	int_pair root_v(root, 0);
	v_list.push_back(root_v);
	while(!v_list.empty())
	{
		int_pair front_v = v_list.pop_front();
		
		if(front_v.y != currentLevel)
		{
			//print forntier size
			//printf("CPU: level %d has %d vertices\n", currentLevel, vertexCount);
			currentLevel ++;
			vertexCount = 0;
		}
		vertexCount ++;	

		//	printf("pop vertex %d\n",front_v.x);
		EdgeNode *edge_ptr = graph.vertexArray[front_v.x].firstEdge;
		while(edge_ptr != NULL)
		{
			int v_id = edge_ptr->edge.dstVertexID;
			if((*level)[v_id] == -1)
			{
				(*level)[v_id] = front_v.y + 1;
				int_pair temp_v(v_id,(*level)[v_id]);
				v_list.push_back(temp_v);
			}
			edge_ptr = edge_ptr->nextEdge;
			//printf("edge_ptr %d\n",(unsigned int)edge_ptr);
		}
	}
	printf("CPU got %d levels\n", currentLevel);
}
//Call This Procedure
//CPUBFS(graph, &test_level, BFS_root);
#define MULTIPLE_RUM
#define RUN_TIMES 1
#define CPU_COUNTER_PART

int main(int argc, char **argv)
{
	
	
	if(argc < 5)
	{
		printf("Usage: Medusa <GPU number> <Number of hops> <Graph file name> <Partition file name> {random root(any str)}\n");
		exit(-1);
	}



	//global configuration
	InitConfig(argv[1], argv[2], NO_MESSAGE, false);



	//print multiple hop statistics
	//AnalyzeReplica(argv[3], argv[4]);

	//exit(-1);

	float t;//timer value
	StopWatchInterface *timer = NULL;
	cutCreateTimer(&timer);
	cutResetTimer(&timer);


	GTGraph gt_graph;
	GraphPartition *gp_array = new GraphPartition[MGLOBAL::num_gpu_to_use];
	gt_graph.get_parition(gp_array, argv[3], argv[4]);


	InitHostDS(gp_array);





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
	
	

#ifdef MULTIPLE_RUM
	for(int run_index = 0; run_index < RUN_TIMES; run_index ++)
	{

#endif

	//<Application Specific!> init levels and set the root	
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		for(int vid = 0; vid < MGLOBAL::gpu_def[i].vertexArray.size; vid ++)
			MGLOBAL::gpu_def[i].vertexArray.level[vid] = -1; 
	}

	//find the physical id of largest_vertex
	int rootVertexID;
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		rootVertexID  = -1;
		HASH_SPACE::hash_map<int, int>::iterator it;
		it = gp_array[i].l2p.begin();
		while(it != gp_array[i].l2p.end())
		{
			if(it->first == largest_vertex)
			{
				rootVertexID = it->second;
				break;
			}
			it ++;
		}

		if(argc == 6)
		{
			srand(time(0));
			rootVertexID = rand32() % MGLOBAL::gpu_def[i].vertexArray.size;
		}

		if(rootVertexID != -1)
		{
			MGLOBAL::gpu_def[i].vertexArray.level[rootVertexID] = 0;
			printf("BSF Root: set %d in partition %d\n", rootVertexID, i);

		}
	}

	gt_file.close();
	//</Application Specific!> init levels and set the root	




	//call this function when all vertex and edge data are ready
#ifdef MULTIPLE_RUM
	if(run_index == 0)
		InitDeviceDS();
	else
		ResetGraphData();
#else
	InitDeviceDS();
#endif



	
	//---------------------------------------------------------------------------------------------------------//
	//execute<BSP loop>
	cutStartTimer(&timer);
	bool exe_temp = true;
	while(exe_temp)
	{	
		DBGPrintf("step = %d ----------------------\n",MGLOBAL::super_step);
		//reset the global GPU side toExecute variable
		ResetToExecute();

		//printf("GRAPH_STORAGE_CPU::super_step = %d\n",GRAPH_STORAGE_CPU::super_step);
		MyCheckErrorMsg("Before Medusa_Exec\n");
		//cutStartTimer(timer);
		Medusa_Exec();

		exe_temp = RetriveToExecute();
		//inc super_step
		(MGLOBAL::super_step) ++;
		
	}

	cutStopTimer(&timer);
#ifdef MULTIPLE_RUM
	}
#endif





#ifdef MULTIPLE_RUM
	t = cutGetTimerValue(&timer)/RUN_TIMES;
#else
	t = cutGetTimerValue(&timer);
#endif
	printf("GPU BFS %.3f ms with %d steps\n",t, MGLOBAL::super_step); 
	//---------------------------------------------------------------------------------------------------------//



	int rootVertexID  = -1;
	HASH_SPACE::hash_map<int, int>::iterator it;
	it = gp_array[0].l2p.begin();
	while(it != gp_array[0].l2p.end())
	{
		if(it->first == largest_vertex)
		{
			rootVertexID = it->second;
			break;
		}
		it ++;
	}
	//<Test CPU counter part>
#ifdef CPU_COUNTER_PART
	if(MGLOBAL::num_gpu_to_use != 1)
	{
		printf("Test CPU counter part, only accept one partition\n");
		exit(-1);
	}
	StopwatchInterface* cpu_timer=NULL;
	cutCreateTimer(&cpu_timer);
	for(int test_count = 0; test_count < RUN_TIMES; test_count ++)
	{
		//init levels and set the root

		for(int i = 0; i < MGLOBAL::gpu_def[0].vertexArray.size; i++)
			MGLOBAL::gpu_def[0].vertexArray.level[i] = MVT_Init_Value;
		MGLOBAL::gpu_def[0].vertexArray.level[rootVertexID] = 0;

		cutStartTimer(&cpu_timer);
		Medusa_Exec_CPU(MGLOBAL::gpu_def[0].vertexArray, MGLOBAL::gpu_def[0].edgeArray);
		cutStopTimer(&cpu_timer);
	}
	t = cutGetAverageTimerValue(&cpu_timer);
	printf("CPU BFS %.3f ms\n", t); 

#endif
	//</Test CPU counter part>

	int testtest = 0;
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		for(int j = 0; j < MGLOBAL::gpu_def[i].replica_index[0]; j ++)
			if(MGLOBAL::gpu_def[i].vertexArray.level[j] != MVT_Init_Value)
				testtest ++;
	}
	printf("CPU reached %d vertices\n", testtest);




	//dump the graph to RAM
	MyCheckErrorMsg("before dump");
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		MGLOBAL::gpu_def[i].d_vertexArray.Dump(MGLOBAL::gpu_def[i].vertexArray);
	}	
	int reached_vertex_num = 0;
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		for(int j = 0; j < MGLOBAL::gpu_def[i].replica_index[0]; j ++)
			if(MGLOBAL::gpu_def[i].vertexArray.level[j] != MVT_Init_Value)
				reached_vertex_num ++;
	}
	printf("GPU Final reached %d vertices\n", reached_vertex_num);

	return 0;
}
