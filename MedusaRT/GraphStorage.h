/****************************************************
* @file GraphStorage.h GPU Graph Storage(vertices/edges/messages) and global variables used by the framework
                                                        are defined here as global variable
* @brief Graph data(vertices/edges/messages) in GPU are defined as structs of arrays in global memory
* @version 0.0
* @author Zhong Jianlong(http://www.jlzhong.com)
* @date 12/31/2010
* Copyleft for non-commercial use only. No warranty.
****************************************************/


#ifndef GraphStorage_H
#define GraphStorage_H

#include "../Algorithm/VertexDataType.h"
#include "../Algorithm/MessageDataType.h"
#include "../Algorithm/EdgeDataType.h"
#include "../MedusaRT/Combiner.h"
#include "../Algorithm/Configuration.h"


namespace GRAPH_STORAGE_CPU{
        //device variable alias, referenced from Host code
        extern D_MessageArray alias_d_messageArray;
        extern D_MessageArray alias_d_messageArrayBuf;
        extern D_EdgeArray alias_d_edgeArray;
        extern D_VertexArray alias_d_vertexArray;
        
        //CPU
        extern VertexArray vertexArray;
        extern EdgeArray edgeArray;

        //const MessageArray messageArray;
        extern cudaDeviceProp device_prop;
        extern int super_step; /* the super step count. starting from 0 */
        extern bool toExecute; /* Corresponding variable of d_toExecute */
        extern Medusa_Combiner com;
        extern MessageMode message_mode;
}


#endif
