#include "MessageDataType.h"

struct Vertex
{
	int edge_count;
	int msg_index;/* the starting index of the vertex's message list(Messages stored in a flat array) */
	MVT distance;
	bool updated;
#ifdef HY
	int edge_index;
#else ifdef AA
	int edge_index;
#endif

};
