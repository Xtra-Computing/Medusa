
#include "../Algorithm/VertexDataType.h"
#include "../Algorithm/EdgeDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../MedusaRT/Combiner.h"
#include "../Algorithm/Configuration.h"


namespace GRAPH_STORAGE_CPU{


	//device variable alias, referenced from Host cod
    D_MessageArray alias_d_messageArray;
    D_MessageArray alias_d_messageArrayBuf;
    D_EdgeArray alias_d_edgeArray;
    D_VertexArray alias_d_vertexArray;
	//CPU
	VertexArray vertexArray;
    EdgeArray edgeArray;
	//const MessageArray messageArray;
	cudaDeviceProp device_prop;
    int super_step; /* the super step count. starting from 0 */
    bool toExecute; /* Corresponding variable of d_toExecute */
    Medusa_Combiner com;
    MessageMode message_mode;


	//gather table for the gathering operation
	int *d_gather_table; //copied from int _gather_table in GPUDef
}
