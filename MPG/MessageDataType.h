/**
* @file MessageDataStructure.h data
* @author Zhong Jianlong
* @date 12/30/2010
* Copyleft for non-commercial use only. No warranty.
*/

#ifndef MESSAGEDATATYPE_H
#define MESSAGEDATATYPE_H

/**
* MessageArray为纯数组结构，不适用MEG
*/

#include <cutil.h>
#include <cuda_runtime.h>
#include "Message.h"

/**
* @brief user defined message type
*/



struct Message
{
	MVT val;

};


struct MessageArray
{
	MVT *val;
	int size;
	MessageArray();
	void resize(int new_size);
};


struct D_MessageArray
{
	MVT *d_val;
	int size;
	void Fill(MessageArray ma);
	void resize(int);
};

/* 暂时没有实现的必要 */
struct MessageList
{

};



#endif