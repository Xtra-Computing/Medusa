/**
* @file utilities.h some GPU side and CPU side utility functions
* @author Zhong Jianlong(zh.jianlong@gmail.com)
* @date 
*/


#ifndef UTILITIES_H
#define UTILITIES_H

/**
* malloc CPU side memory
* @param [out] ptr return memory pointer to prt
* @param [in] size size of the memory to malloc
*/
void CPUMalloc(void **ptr, int size);


/**
* malloc GPU side memory
* @param [out] ptr return memory pointer to prt
* @param [in] size size of the memory to malloc
*/
void GPUMalloc(void **ptr, int size);

/**
* generate random number in the range of 0 ~ 2^32 - 1
* @param [in] max the maximum value of the generated number(inclusive)
* @note unsigned int type should be at least 32bits
*/
unsigned int rand32();

/**
* init the GPU execution environment. The property of the device is store as global variable.
*
* @param []	
* @return	
* @note	
*
*/
void InitCUDA();



/*
 *	Print usage information.
 *
 */
void Usage();

#endif
