#include "MessageDataType.h"
#include "Configuration.h"

struct Vertex
{
	int edge_count;
	int pg_edge_num;
	MVT level;
	int msg_index;/* the starting index of the vertex's message list(Messages stored in a flat array) */

#ifdef HY
	int edge_index;
#else ifdef AA
	int edge_index;
#endif

};
