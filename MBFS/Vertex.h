
#include "MessageDataType.h"
#include "Configuration.h"

struct Vertex
{
	int edge_count;
	MVT level;
#ifdef HY
	int edge_index;
#else ifdef AA
	int edge_index;
#endif

	
};
