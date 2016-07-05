/****************************************************
* @file 
* @brief 
* @version
* @author Zhong Jianlong(http://www.jlzhong.com)
* @date 2011/01/12
* Copyleft for non-commercial use only. No warranty.
****************************************************/

#ifndef VIS_GLOBAL_VARIABLES
#define VIS_GLOBAL_VARIABLES
#ifdef _WIN32
#include <Windows.h>
#endif

#include <cuda_runtime.h>
//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
//#include <cutil.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include "../Compatibility/Compatability.h"
#include <cuda_gl_interop.h>

extern GLuint vertex_vbo;
extern GLuint vertex_color_vbo;
extern struct cudaGraphicsResource* vertex_vbo_cuda;
extern struct cudaGraphicsResource* vertex_color_vbo_cuda;

extern GLuint edge_vbo;
extern GLuint edge_color_vbo;
extern struct cudaGraphicsResource* edge_vbo_cuda;
extern struct cudaGraphicsResource* edge_color_vbo_cuda;

/**
* render all the vertices
*
* @param   - 
* @return	
* @note	
*
*/
void RenderVertex();

/**
* render all the edges
*
* @param   - 
* @return	
* @note	
*
*/
void RenderEdges();


/**
* detach resources from CUDA and start OpenGL rendering
*
* @param   - 
* @return	
* @note	
*
*/
void StartRendering();



/**
*
*
* @param   - 
* @return	
* @note	
*
*/
void StartMainLoop();

/**
*
*
* @param   - 
* @return	
* @note	
*
*/
void InitInterOp();

#endif
