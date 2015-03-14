#include "../Algorithm/VertexDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../Algorithm/EdgeDataType.h"
#include "../MedusaRT/Combiner.h"
#include "../Algorithm/Configuration.h"
#include "../MultipleGPU/GPUDef.h"
#include "../MultipleGPU/MultiGraphStorage.h"
#include <pthread.h>


namespace MGLOBAL{
	//per GPU specific CPU side data structures
	GPUDef *gpu_def;
	//shared variables by multiple GPUs
	int super_step; /* the super step count. starting from 0 */
	bool toExecute; /* Corresponding variable of d_toExecute */
	Medusa_Combiner com;
	CUDPPDatatype combiner_datatype;
	CUDPPOperator combiner_operator;
	MessageMode message_mode;
	MedusaDS graph_ds_type;
	int num_gpu_to_use;
	int current_device = 0;
	MVT *original_buffer;
	int original_buffer_size;
	pthread_t replica_update_thread;
	int max_hop;
	int *total_edge_count;
}

void medusaSetDevice(int device_id)
{
	if(device_id > MGLOBAL::num_gpu_to_use)
	{
		printf("Set device error\n");
		exit(-1);
	}
	if(device_id != MGLOBAL::current_device)
	{
		MGLOBAL::current_device = device_id;
		if(cudaSetDevice(device_id)!=cudaSuccess)
		{
			printf("Switch to device %d\n",device_id);
			exit(-1);
		}
	}
}
