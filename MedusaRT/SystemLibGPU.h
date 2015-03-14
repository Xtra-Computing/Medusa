/****************************************************
* @file SystemLibGPU.h
* @brief System library calls. Users can call these functions when writing the __device__ functors
* @version
* @author Zhong Jianlong(http://www.jlzhong.com)
* @date 12/30/2010
* Copyleft for non-commercial use only. No warranty.
****************************************************/

#ifndef SYSTEMLIBGPU
#define SYSTEMLIBGPU


__device__ void Medusa_Break()
{
	GRAPH_STORAGE_GPU::d_toExecute = false;
}

__device__ void Medusa_Continue()
{
	GRAPH_STORAGE_GPU::d_toExecute = true;
}


#endif