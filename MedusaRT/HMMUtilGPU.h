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

#ifndef HMMUTILGPU_H
#define HMMUTILGPU_H

#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>


namespace HMM_GPU{

	const int MIN_ALIGN=4;

	// =====================================
	// device functions
	// =====================================
	template<class T>
	__device__ T HMMAlign(T n, int b){
		return ((int)n&(b-1))==NULL ? n : n+b-((int)n&(b-1));
	}

	template<class T>
	__device__ T minAlign(T n){
		return ((int)n&(MIN_ALIGN-1))==NULL ? n : n+MIN_ALIGN-((int)n&(MIN_ALIGN-1));
	}

	template<class T1, class T2, class T3>
	__device__ bool CAS64(T1 * addr, T2 old_val, T3 new_val){
		return *(unsigned long long *)(&old_val)==atomicCAS((unsigned long long *)addr, *(unsigned long long *)(&old_val), *(unsigned long long *)(&new_val));
	}

	template<class T1, class T2, class T3>
	__device__ bool CAS32(T1 * addr, T2 old_val, T3 new_val){
		return *(int *)(&old_val)==atomicCAS((int *)addr, *(int *)(&old_val), *(int *)(&new_val));
	}

	template<class T1, class T2, class T3>
	__device__ bool CASPTR(T1 * addr, T2 old_val, T3 new_val){
#ifdef LONG_PTR
		return CAS64(addr, old_val, new_val);
#else
		return CAS32(addr, old_val, new_val);
#endif
	}

	template<class T1, class T2>
	__device__ int ADD32(T1 * addr, T2 val){
		return atomicAdd((int *)addr, *(int *)(&val));
	}

	__device__ int getThreadID(){
		//	int block_id=blockIdx.y*gridDim.x+blockIdx.x;
		//	int blockSize=blockDim.z*blockDim.y*blockDim.x;
		//	int thread_id=threadIdx.z*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
		//	return block_id*blockSize+thread_id;
		return blockIdx.x*blockDim.x+threadIdx.x;
	}

	__device__ int getNumThreads(){
		return (gridDim.y*gridDim.x)*(blockDim.x*blockDim.y*blockDim.z);
	}

	__device__ void copyVal(void * dst, const void * src, unsigned int size){
		char * d=(char*)dst;
		const char * s=(const char *)src;
		for(int i=0;i<size;i++)
			d[i]=s[i];
	}

	//=====================================
	// host functions
	//=====================================
	template<class T>
	__host__ T * createDeviceVar(const T & h_var){
		T * d_addr;
		cudaMalloc((void **)&d_addr, sizeof(T));
		cudaMemcpy(d_addr, &h_var, sizeof(T),cudaMemcpyHostToDevice);
		return d_addr;
	}

	template<class T>
	__host__ T downloadDeviceVar(T * d_ptr){
		T temp;
		cudaMemcpy(&temp, d_ptr, sizeof(T),cudaMemcpyDeviceToHost);
		return temp;
	}

#  define CUT_CHECK_ERROR(errorMessage) do {				 \
	cudaError_t err = cudaGetLastError();				    \
	if( cudaSuccess != err) {						\
	fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
	errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);						  \
	}									\
	err = cudaThreadSynchronize();					   \
	if( cudaSuccess != err) {						\
	fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
	errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
	exit(EXIT_FAILURE);						  \
	} } while (0)

#  define CE(call) do {                                \
	call;CUT_CHECK_ERROR("------- Error ------\n"); \
	} while (0)


	static unsigned int allocated_mem_size[8];
	__host__ void * dMalloc(unsigned int size){
		void * ptr;
		CE(cudaMalloc((void**)&ptr, size));
		int gpu_num=0;
		CE(cudaGetDevice(&gpu_num));
		allocated_mem_size[gpu_num]+=size;
		return ptr;
	}

	__host__ void dFree(void * ptr, unsigned int size){
		CE(cudaFree(ptr));
		int gpu_num=0;
		CE(cudaGetDevice(&gpu_num));
		allocated_mem_size[gpu_num]-=size;
	}

	__host__ void * hMalloc(unsigned int size){
		void * ptr;
		CE(cudaMallocHost((void**)&ptr, size));
		return ptr;
	}

#define memcpyToSymbol(symbol, src) CE(cudaMemcpyToSymbol(symbol, src, sizeof(symbol), 0, cudaMemcpyHostToDevice));

#define memcpyFromSymbol(dst, symbol) CE(cudaMemcpyFromSymbol(dst, symbol, sizeof(symbol), 0, cudaMemcpyDeviceToHost));

	__host__ void memcpyD2H(void * dst, const void * src, unsigned int size){
		CE(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
	}

	__host__ void memcpyH2D(void * dst, const void * src, unsigned int size){
		CE(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
	}

	__host__ void dMemset(void * ptr, char c, unsigned int size){
		CE(cudaMemset(ptr,c,size));
	}

	__host__ unsigned int getGlobalMemSize(int deviceNum){
		cudaDeviceProp dp;
		CE(cudaGetDeviceProperties(&dp, deviceNum));
		return dp.totalGlobalMem;	// reserve 10MB for other app
	}

	__host__ unsigned int getAllocatableMemSize(int deviceNum){
		int gpu_num=0;
		CE(cudaGetDevice(&gpu_num));
		return getGlobalMemSize(gpu_num)-allocated_mem_size[gpu_num];
	}

};
#endif
