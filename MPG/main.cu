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
#include "../MultipleGPU/PartitionManager.h"
#include "../MultipleGPU/MultiUtilities.h"
#include <fstream>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline_runtime.h>


#include "../Tools/ReplicaNumberAnalysis.h"



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
	InitConfig(argv[1], argv[2], EDGE_MESSAGE, true);
	MGLOBAL::combiner_datatype = CUDPP_FLOAT;
	MGLOBAL::combiner_operator = CUDPP_ADD;
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

	InitHostDS(gp_array);
	CheckOutgoingEdge(gp_array);
	printf("InitHostDS Done\n");



	/* <algorithm specific initialization>  */
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);	
		//init PageRank values
		for(int vertex_index = 0; vertex_index < MGLOBAL::gpu_def[i].vertexArray.size; vertex_index++)
			MGLOBAL::gpu_def[i].vertexArray.level[vertex_index] = 1.0;
	}


	int total_vertex_num = 0;
	int total_edge_num = 0;
	ifstream gt_file(argv[3]);
	ifstream partition_file(argv[4]);
	if(!gt_file.good())
	{
		printf("open GT Graph File failed\n");
		exit(-1);
	}



	//not that multiple-GPU support for edge list is not implemented ye, thus MEG can be only
	//used for single-GPU case.
	if(MGLOBAL::num_gpu_to_use > 1)
	{
		char line[1024];
		char first_ch;
		while(gt_file.get(first_ch))
		{
			if(first_ch == 'p')
			{
				string temp;
				gt_file>>temp>>total_vertex_num>>total_edge_num;
				gt_file.getline(line, 1024);//eat the line break
				break;
			}
			gt_file.getline(line, 1024);//eat the line break
		}
		int *pg_edge_num = (int *)malloc(sizeof(int)*(total_vertex_num + 1));
		memset(pg_edge_num, 0, sizeof(int)*(total_vertex_num + 1));

		int src_id, dst_id;
		float edge_weight;

		while(gt_file.get(first_ch))
		{
			if(first_ch == 'a')
			{
				gt_file>>src_id>>dst_id>>edge_weight;
				pg_edge_num[src_id] ++;
			}
		}

		for(int vid = 1; vid <= total_vertex_num; vid ++)
		{
			for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
			{
				if(gp_array[i].l2p.find(vid) != gp_array[i].l2p.end())
					MGLOBAL::gpu_def[i].vertexArray.pg_edge_num[gp_array[i].l2p[vid]] = pg_edge_num[vid];
			}
		}
	}
	else
	{
		//for one GPU execution with no replication, just check the GraphIR data structure
		for(int vpid = 0; vpid < gp_array[0].graph.vertexNum; vpid ++)
			MGLOBAL::gpu_def[0].vertexArray.pg_edge_num[vpid] = gp_array[0].graph.vertexArray[vpid].vertex.edge_count;
	}
	/*  </algorithm specific initialization>  */




	InitDeviceDS();

	//---------------------------------------------------------------------------------------------------------//
	//execute<PG loop>
	bool exe_temp = true;
	cutResetTimer(timer);


	while(exe_temp)
	{

		//reset the global GPU side toExecute variable

//		printf("step = %d ----------------------------------------------------------\n",MGLOBAL::super_step);

		cutStartTimer(timer);
		Medusa_Exec();
		cutStopTimer(timer);
		
	
		exe_temp = RetriveToExecute();
		//inc super_step
		(MGLOBAL::super_step) ++;


		//loop termination condition
		if((MGLOBAL::super_step) >= 100)
			break;

	}


	t = cutGetTimerValue(timer);
	printf("GPU PG %.3f ms\n",t); 


	//check results
	cutilCheckMsg("before dump");
	double total_pg_value = 0;
	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		medusaSetDevice(i);
		MGLOBAL::gpu_def[i].d_vertexArray.Dump(MGLOBAL::gpu_def[i].vertexArray);
	}	

	for(int i = 0; i < MGLOBAL::num_gpu_to_use; i ++)
	{
		for(int j = 0; j < MGLOBAL::gpu_def[i].replica_index[0]; j ++)
			total_pg_value += MGLOBAL::gpu_def[i].vertexArray.level[j];
	}
	printf("GPU total pg value = %f\n", total_pg_value);



#ifdef CPU_COUNTER_PART
	

	Init_CPU_Medusa(MGLOBAL::gpu_def[0].edgeArray);

	
	cutResetTimer(timer);
	for(int i = 0; i < RUN_TIMES; i ++)
	{

		//<Algorithm specific initialization>
		for(int vertex_index = 0; vertex_index < MGLOBAL::gpu_def[0].vertexArray.size; vertex_index++)
			MGLOBAL::gpu_def[0].vertexArray.level[vertex_index] = 1.0;
		//</Algorithm specific initialization>


		cutStartTimer(timer);
		Medusa_Exec_CPU(MGLOBAL::gpu_def[0].vertexArray, MGLOBAL::gpu_def[0].edgeArray);
		cutStopTimer(timer);
	}


	total_pg_value = 0.0;
	for(int j = 0; j < MGLOBAL::gpu_def[0].replica_index[0]; j ++)
		total_pg_value += MGLOBAL::gpu_def[0].vertexArray.level[j];


	t = cutGetTimerValue(timer);
	printf("CPU PG %.3f ms\n",t/RUN_TIMES); 

	printf("CPU total pg value = %f\n", total_pg_value);

#endif


	return 0;
}
