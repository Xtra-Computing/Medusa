#ifndef MultiGraphStorage_H
#define MultiGraphStorage_H

#include "../Algorithm/VertexDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../Algorithm/EdgeDataType.h"
#include "../MedusaRT/Combiner.h"
#include "../Algorithm/Configuration.h"
#include "../MultipleGPU/GPUDef.h"

#include <pthread.h>
#include <cudpp.h>

namespace MGLOBAL{

	//per GPU specific CPU side data structures
	extern GPUDef *gpu_def;

	//shared variables by multiple GPUs
	extern int super_step; /* the super step count. starting from 0 */
	extern bool toExecute; /* Corresponding variable of d_toExecute */
	extern Medusa_Combiner com;
	extern CUDPPDatatype combiner_datatype;
	extern CUDPPOperator combiner_operator;
	extern MessageMode message_mode;
	extern MedusaDS graph_ds_type;
	extern int num_gpu_to_use;


	extern int current_device;
	extern MVT *original_buffer;
	extern int original_buffer_size;
	extern pthread_t replica_update_thread;
	extern int max_hop;


	//for HY, record real number of edges
	extern int *total_edge_count;
}

void medusaSetDevice(int device_id);


#endif
