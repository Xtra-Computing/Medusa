#ifndef CONFIGURATION_H
#define CONFIGURATION_H


#define CUDPP_2_0
#define MEDUSA_MAX 1000000000

#define MyRelease

#ifndef MyRelease
#define MyCheckErrorMsg(msg) cutilCheckMsg(msg)
#else
#define MyCheckErrorMsg(msg)
#endif


#ifndef MyRelease
#define MySafeCall(foo) CUDA_SAFE_CALL(foo)
#else
#define MySafeCall(foo) foo
#endif


#ifndef MyRelease
#define DBGPrintf printf
#else
#define DBGPrintf //
#endif



/*
 *	Define APP name
 */
#define SSSP


/*
 * Define if need multiple GPU execution
 */
//#define MULTIGPU


/**
* Message Mode
* Edge_MESSAGE -- Each edge is allocated a message buffer
* VERTEX_MESSAGE -- Each vertex is allocated a message buffer
* NO_MESSAGE -- The application doesn't need to use message
*/

enum MessageMode{EDGE_MESSAGE, VERTEX_MESSAGE, NO_MESSAGE};
enum MedusaDS{DS_MEG, DS_AA, DS_HY, DS_ELL};
/**
* enable or disable visualization
*/
//#define VIS
/**
* enable BFS mark and BFS filtering
*/
//#define BFS_MARK
//#define BFS_FILTERING



/**
* enable EDGE_LIST_API
*/
//#define ENABLE_EDGE_LIST
//#define MEG
#define AA
//#define HY
//#define ELL

/**
* for testing different representations of the graph
*/
//#define TEST_DS



/*
 * Enable the queue structure.
 * Define which kind of enqueue method will be used
 */
//#define WARPBASED
//#define SCANBASED


/*
 * Define what kind of queue will be used
*/
//#define EAlpha
//#define EBeta
//#define VAlpha
//#define VBeta

#endif
