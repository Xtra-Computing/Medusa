#include "../Algorithm/Configuration.h"
#ifdef VIS

#include <GL/glew.h>
#include <GL/glut.h>
#include "GraphRenderingAPI.h"
#include "GraphStorage.h"
#include "OpenglSettings.h"
#include "CUDAOpenglInterop.h"
#include "GraphStorage.h"
#include "Configuration.h"

GLuint vertex_vbo;
GLuint vertex_color_vbo;
struct cudaGraphicsResource* vertex_vbo_cuda;
struct cudaGraphicsResource* vertex_color_vbo_cuda;




GLuint edge_vbo;
GLuint edge_color_vbo;
struct cudaGraphicsResource* edge_vbo_cuda;
struct cudaGraphicsResource* edge_color_vbo_cuda;

/* buffer pointers */
float2 *d_vertex_quad;
uchar4 *d_color;




__global__ void DrawNodes(float2 *d_vertex_quad, int *d_radius, float2* d_pos,const int VertexNum)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= VertexNum)
		return;
	float2 cent = d_pos[tid];
	int radius = d_radius[tid];
	d_vertex_quad[tid*4] = make_float2(cent.x - radius, cent.y + radius - 1);
	d_vertex_quad[tid*4 + 1] = make_float2(cent.x + radius - 1, cent.y + radius - 1);
	d_vertex_quad[tid*4 + 2] = make_float2(cent.x + radius - 1, cent.y - radius);
	d_vertex_quad[tid*4 + 3] = make_float2(cent.x - radius, cent.y - radius);

}

void RenderVertex()
{
	size_t num_bytes;
	//vertex coordinates
	cudaGraphicsMapResources(1, &vertex_vbo_cuda, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_vertex_quad, &num_bytes, vertex_vbo_cuda);
	//printf("accessible bytes = %d\n",num_bytes);


	// execute the kernel
	int blockX = 256;
	int gridX = GRAPH_STORAGE_CPU::alias_d_vertexArray.size/blockX;
	if(GRAPH_STORAGE_CPU::alias_d_vertexArray.size%blockX)
		gridX ++;

	DrawNodes<<<gridX,blockX>>>( d_vertex_quad, GRAPH_STORAGE_CPU::alias_d_vertexArray.d_radius, GRAPH_STORAGE_CPU::alias_d_vertexArray.d_pos, GRAPH_STORAGE_CPU::alias_d_vertexArray.size);
	cudaThreadSynchronize();
	MyCheckErrorMsg("DrawNodes");
	
	
}

__global__ void DrawEdges(float2 *d_edge_points_coord, int *d_edge_src_vid, int *d_edge_dst_vid, float2 *d_v_pos, int edge_num)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid >= edge_num)
		return;
//	printf("src = %d dst = %d\n",d_edge_src_vid[tid],d_edge_dst_vid[tid]);
	d_edge_points_coord[tid*2] = d_v_pos[d_edge_src_vid[tid]];
	d_edge_points_coord[tid*2 + 1] = d_v_pos[d_edge_dst_vid[tid]];

/*	should be set in the edge processor
	d_edge_points_color[tid*2] = make_uchar4(128,0,0);
	d_edge_points_color[tid*2 + 1] = make_uchar4(128,0,0);
*/

}


void RenderEdges()
{
	
	float2 *d_edge_points_coord;
	uchar4 *d_edge_points_color;
	size_t num_bytes;

	
	
	cudaGraphicsMapResources(1, &edge_vbo_cuda, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_edge_points_coord, &num_bytes, edge_vbo_cuda);

	

	int blockX = 256;
	int gridX = GRAPH_STORAGE_CPU::alias_d_edgeArray.size/blockX;
	if(GRAPH_STORAGE_CPU::alias_d_edgeArray.size%blockX)
		gridX ++;
	DrawEdges<<<gridX, blockX>>>(d_edge_points_coord, GRAPH_STORAGE_CPU::alias_d_edgeArray.d_srcVertexID, GRAPH_STORAGE_CPU::alias_d_edgeArray.d_dstVertexID, GRAPH_STORAGE_CPU::alias_d_vertexArray.d_pos, GRAPH_STORAGE_CPU::alias_d_edgeArray.size);
	cudaThreadSynchronize();
	MyCheckErrorMsg("DrawEdges");

}


/**
* unmap the resources
*
* @param   - 
* @return	
* @note	
*
*/
void StartRendering()
{
	RenderVertex();
	cudaGraphicsUnmapResources(1, &vertex_vbo_cuda, 0); 
	cudaGraphicsUnmapResources(1, &vertex_color_vbo_cuda, 0); 

	RenderEdges();
	cudaGraphicsUnmapResources(1, &edge_vbo_cuda, 0); 
	cudaGraphicsUnmapResources(1, &edge_color_vbo_cuda, 0);

	glutMainLoop();
}

void InitInterOp()
{
	createVBO(&vertex_vbo, &vertex_vbo_cuda, sizeof(float2)*GRAPH_STORAGE_CPU::alias_d_vertexArray.size*4);
	createVBO(&edge_vbo, &edge_vbo_cuda, sizeof(float2)*GRAPH_STORAGE_CPU::alias_d_edgeArray.size*2);//every edge two end points
}

void StartMainLoop()
{
	glutMainLoop();
}

#endif
