#ifndef WEQUEUE_H
#define WEQUEUE_H
#include "../MultipleGPU/MultiGraphStorage.h"

//-------------Queue Ptr related functions-------------------
void SetEdgeQueuePtr( int _ptrValue);
int GetEdgeQueuePtr();
void SetVertexQueuePtr( int _ptrValue);
int GetVertexQueuePtr();

void SwapEdgeQueue();
void SwapVertexQueue();

#endif
