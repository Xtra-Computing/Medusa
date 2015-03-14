/**
* @date 2012/02/07
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


struct MessageList
{

};



#endif