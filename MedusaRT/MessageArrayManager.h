/****************************************************
* @file 
* @brief 
* @version
* @author Zhong Jianlong(http://www.jlzhong.com)
* @date 2011/01/02
* Copyleft for non-commercial use only. No warranty.
****************************************************/
#ifndef MESSAGEARRAYMANAGER_H
#define MESSAGEARRAYMANAGER_H
#include <cudpp.h>
#include "../Algorithm/MessageDataType.h"

__global__ void MsgArrayInit(MVT init_val, int total_thread_count);


/**
* interface function for initialize the message array
*
* @param []	
* @return	
* @note	
*
*/
void InitMessageBuffer(Message init_val);




#endif