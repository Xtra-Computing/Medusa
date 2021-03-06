#ifndef CUDAOPENGLINTEROP_H
#define CUDAOPENGLINTEROP_H

#ifdef _WIN32
#include <Windows.h>
#endif

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glext.h>

#include <cuda_runtime.h>
//#include <cutil.h>
//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include "../Compatibility/Compatability.h"

#include <cuda_gl_interop.h>
#include "GraphRenderingAPI.h"

void createVBO(GLuint* vbo, struct cudaGraphicsResource** vbo_CUDA, int size);
void deleteVBO(GLuint* vbo);
void InitGraphics(int argc, char **argv);


#endif
