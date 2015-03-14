/****************************************************
* @file 
* @brief 
* @version
* @author Zhong Jianlong(http://www.jlzhong.com)
* @date 2011/01/13
* Copyleft for non-commercial use only. No warranty.
****************************************************/
#include "../Algorithm/Configuration.h"
#ifdef VIS
#include "CUDAOpenglInterop.h"
#include "CUDAOpenglInterop.h"



void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_CUDA, int size)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(vbo_CUDA, *vbo, cudaGraphicsMapFlagsWriteDiscard);

}



void deleteVBO(GLuint* vbo)
{
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	cudaGLUnregisterBufferObject(*vbo);

	*vbo = NULL;
}

#endif
