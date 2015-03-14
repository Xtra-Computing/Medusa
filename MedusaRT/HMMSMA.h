/***************************************************************************

MapCG: MapReduce Framework for CPU & GPU

Copyright (C) 2010, Chuntao HONG (Chuntao.Hong@gmail.com).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

**************************************************************************/

#ifndef HMMSMAGPU_H
#define HMMSMAGPU_H

#ifdef SMA_DEBUG
#include <stdio.h>
#endif

//#include "DLog.h"
#include "HMMUtilGPU.h"

namespace HMM_GPU{

	/*********************************
	parameters that should be changed
	by the programmer
	**********************************
	*/
	const unsigned int THREAD_NUM=256;	// number of threads in a block
	const unsigned int WARP_SIZE=32;

	/*********************************
	type defines and pre-defines
	**********************************
	*/
	const unsigned int PAGE_SIZE=1024;
	const unsigned int ALIGN_SIZE=4;
	const unsigned int WARP_NUM=THREAD_NUM/WARP_SIZE;

	/*********************************
	data structures used
	**********************************
	*/
	__constant__ char * sma_global_pool;			// memory pool
	__constant__ unsigned int sma_global_capacity;		// capacity of the memory pool
	__device__ unsigned int sma_global_offset;		// the offset of the pool

	__shared__ volatile char * sma_local_page[WARP_NUM];		// the local page in shared memory, one for each warp
	__shared__ volatile unsigned int sma_local_offset[WARP_NUM];	// the current offset for local page in shared memory, one for each warp

	/*********************************
	function implementation
	**********************************
	*/

	//============================
	// tools
	__device__ unsigned int SMA_RoundToPageSize(unsigned int size){
		return (size+PAGE_SIZE-1)&(~(PAGE_SIZE-1));
	}
	__device__ unsigned int SMA_RoundToAlign(unsigned int size){
		return (size+ALIGN_SIZE-1)&(~(ALIGN_SIZE-1));
	}
	__device__ unsigned int SMA_GetThreadID(){
		return threadIdx.z*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
	}

	//============================
	// initialization and destruction
	__host__ void SMA_Init(unsigned int size){
#ifdef SMA_DEBUG
		printf("SMA_Init(%d)\n",size);
#endif
		// malloc the memory pool
		char * d_pool=(char *)dMalloc(size);
#ifdef SMA_DEBUG
		printf("pool address at init: %p\n",d_pool);
#endif
		// copy the variables to constant memory
		memcpyToSymbol(HMM_GPU::sma_global_capacity, &size);
		memcpyToSymbol(HMM_GPU::sma_global_pool, &d_pool);
		unsigned int offset=0;
		memcpyToSymbol(HMM_GPU::sma_global_offset, &offset);	// set the offset to 0
	}

	__host__ void SMA_Destroy(){
		// get the pool varaible
		char * pool;
		memcpyFromSymbol(&pool, sma_global_pool);
		unsigned int size;
		memcpyFromSymbol(&size, sma_global_capacity);
#ifdef SMA_DEBUG
		printf("pool address at destroy: %p\n", pool);
		unsigned int offset;
		memcpyFromSymbol(&offset, sma_global_offset);
		printf("offset at destroy: %d\n", offset);
#endif
		// free the memory
		dFree(pool, size);
	}

	//==========================
	// malloc large block from global pool
	__device__ void * SMA_MallocLargeBlock(unsigned int size){
		unsigned int offset=atomicAdd(&sma_global_offset, size);
		if(offset+size>=sma_global_capacity)
			return NULL;
		else
			return sma_global_pool+offset;
	}

	//=========================
	// malloc a page from global pool
	__device__ char * SMA_MallocPage(){
		unsigned int offset=atomicAdd(&sma_global_offset, PAGE_SIZE);
		if(offset+PAGE_SIZE>=sma_global_capacity)
			return NULL;
		else
			return sma_global_pool+offset;
	}

	//=========================
	// malloc from local page
	__device__ void * SMA_MallocFromLocalPage(unsigned int size){
		int threadID=SMA_GetThreadID();
		int warp=threadID/WARP_SIZE;
		char * page_start;
		unsigned int offset;
		bool ret=false;
		while(!ret){
			page_start=(char*)sma_local_page[warp];
			offset=atomicAdd((unsigned int *)(&(sma_local_offset[warp])), size);
			// if successful, break and return
			if(offset+size<=PAGE_SIZE){
				ret=true;
			}
			else{
				// if not enough space, the first failed thread get a new one
				if(offset<=PAGE_SIZE && offset+size>PAGE_SIZE){
					page_start=SMA_MallocPage();
					offset=0;
					// this thread gets memory directly and return
					sma_local_page[warp]=page_start;
					sma_local_offset[warp]=size;
					ret=true;;
				}
				// the other failed threads will try again
			}
		}
		return page_start+offset;
	}

	//=========================
	// setup the variables in shared memory
	//  program should invoke this function 
	//  at the beginning of each __global__ invoke
	//    setup the local pages
	//    return 0 if successful
	//    -1 if fail
	__device__ int SMA_Start_Kernel(){
		int threadID=SMA_GetThreadID();
		int warp=threadID/WARP_SIZE;
		int retval=0;
		//only the first thread in the warp need to do this
		if(threadID%WARP_SIZE ==0){
			char * p=(char *)SMA_MallocPage();
			if(p!=NULL){
				sma_local_page[warp]=p;
				sma_local_offset[warp]=0;
				retval=0;
			}
			else
				retval=-1;
		}
		__syncthreads();
		return retval;
	}

	__device__ void * SMA_Malloc(unsigned int size){
		//DLog<<sma_global_offset;
		if(size>PAGE_SIZE)
			// if large block, get it from the memory pool directly
			return SMA_MallocLargeBlock( SMA_RoundToAlign(size) );
		else
			// for small blocks, try to malloc from local page	
			return SMA_MallocFromLocalPage( SMA_RoundToAlign(size) );
	}

	__device__ void SMA_Free(void * p){

	}

	__host__ void SMA_Get_Status(void * & ptr, unsigned int & offset){
		memcpyFromSymbol(&ptr, sma_global_pool);
		memcpyFromSymbol(&offset, sma_global_offset);
	}

	__host__ void * SMA_Malloc_From_Host(unsigned int size){
		unsigned int aligned_size=(size+PAGE_SIZE-1)&(~(PAGE_SIZE-1));
		void * ptr;
		unsigned int offset;
		unsigned int capacity;
		SMA_Get_Status(ptr, offset);
		memcpyFromSymbol(&capacity, sma_global_capacity);
		if(offset+aligned_size < capacity){
			void * ret_val=(char *)ptr+offset;
			offset+=aligned_size;
			memcpyToSymbol(sma_global_offset, &offset);
			return ret_val;
		}
		else{
			return NULL;
		}
	}
};
#endif
