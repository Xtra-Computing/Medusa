#include "../Algorithm/Configuration.h"
#ifdef VIS

#include <GL/glew.h>
#include <GL/glut.h>
#include "OpenglSettings.h"
#include "GraphRenderingAPI.h"
#include "stdlib.h"
#include "stdio.h"
#include "GraphStorage.h"
#include "../MedusaRT/PublicAPI.h"
#include "../MedusaRT/GraphReader.h"
#include "../MedusaRT/GraphRenderingAPI.h"
#include "../Algorithm/CPUFunctorHoster.h"

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -1000;
float translate_x = 0;
float translate_y = 0;
unsigned int timer;
VertexArray varr;
EdgeArray ea;


void processNormalKeys(unsigned char key, int x, int y) {

	float step;
	if(translate_z > 10000 | translate_z < -10000)
		step = 1000;
	else if(translate_z > 1000 | translate_z < -1000)
		step = 100;
	else if(translate_z > 100 | translate_z < -100)
		step = 10;
	else
		step = 1;

	if ((key == 's') | (key == 'S')) 
		translate_z -= step;
	else if ((key == 'w') | (key == 'W')) 
		translate_z += step;
	//printf("translate_z = %f\n",translate_z);
	glutPostRedisplay();
}
void processSpecialKeys(int key, int x, int y) {


	float step_x;
	if(translate_z > 10000 | translate_z < -10000)
		step_x = 1000;
	else if(translate_z > 1000 | translate_z < -1000)
		step_x = 100;
	else if(translate_z > 100 | translate_z < -100)
		step_x = 10;
	else
		step_x = 1;

	float step_y;
	if(translate_z > 10000 | translate_z < -10000)
		step_y = 1000;
	else if(translate_z > 1000 | translate_z < -1000)
		step_y = 100;
	else if(translate_z > 100 | translate_z < -100)
		step_y = 10;
	else
		step_y = 1;

	if (key == GLUT_KEY_UP) 
		translate_y += step_y;
	else if (key == GLUT_KEY_DOWN) 
		translate_y -= step_y;
	else if (key == GLUT_KEY_LEFT) 
		translate_x -= step_x;
	else if (key == GLUT_KEY_RIGHT) 
		translate_x += step_x;

	glutPostRedisplay();
}



void motion(int x, int y)
{
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	if (mouse_buttons & 1) {
		rotate_x += dy * 0.2;
		rotate_y += dx * 0.2;
	} else if (mouse_buttons & 4) {
		translate_z += dy * 0.01;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}


// Mouse event handlers for GLUT
void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1<<button;
	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}
	printf("mouse");
	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}


void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if(h == 0)
		h = 1;

	float ratio = 1.0* w / h;

	// Reset the coordinate system before modifying
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45,ratio,1,1000000);
	glMatrixMode(GL_MODELVIEW);


}

#define CPU_EXEC

void renderScene(void) {
	cutStartTimer(timer);
	


#ifdef CPU_EXEC
	Medusa_Exec_CPU(varr, ea);
	CUDA_SAFE_CALL(cudaMemcpy(GRAPH_STORAGE_CPU::alias_d_vertexArray.d_pos, varr.pos, sizeof(float2)*GRAPH_STORAGE_CPU::alias_d_vertexArray.size, cudaMemcpyHostToDevice));
	cutStopTimer(timer);
#else
	Medusa_Exec();
	cutStopTimer(timer);
#endif


//	glRasterPos2i(0, 0);
//	glColor4f(0.0f, 0.0f, 1.0f, 1.0f);
//	glutBitmapCharacter(GLUT_BITMAP_9_BY_15,'A');



	
	float t = cutGetTimerValue(timer);
	printf("elapsed time %f\n",t);
	RenderVertex();
	cudaGraphicsUnmapResources(1, &vertex_vbo_cuda, 0); 
	


	RenderEdges();
	cudaGraphicsUnmapResources(1, &edge_vbo_cuda, 0); 





	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(translate_x, translate_y, translate_z);


	//enable color blending
	glEnable(GL_BLEND);
	

	glBindBuffer(GL_ARRAY_BUFFER, edge_vbo);
	glVertexPointer(2, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, edge_color_vbo);
	glColorPointer(4, GL_UNSIGNED_BYTE, 4, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	//glLineWidth(3);
	glDrawArrays(GL_LINES, 0, 2*GRAPH_STORAGE_CPU::alias_d_edgeArray.size*2);


	glDisable(GL_BLEND);
	glBindBuffer(GL_ARRAY_BUFFER, vertex_vbo);
	glVertexPointer(2, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, vertex_color_vbo);
	glColorPointer(4, GL_UNSIGNED_BYTE, 4, 0);
	glEnableClientState(GL_COLOR_ARRAY);

	/*glColor3ub(255,0,255);*/
	glDrawArrays(GL_QUADS, 0, GRAPH_STORAGE_CPU::alias_d_vertexArray.size*4);


	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

#ifdef NAMES
	char *str = "Jiawei Han";
	int n = strlen(str);
	glRasterPos2i(varr.pos[5].x + 5,varr.pos[5].y - 5);
	for (int i = 0; i < n; i++)
			glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));
	
	str = "Elisa Bertino";
	n = strlen(str);
	glRasterPos2i(varr.pos[16].x + 5,varr.pos[16].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));

	str = "Philip S. Yu";
	n = strlen(str);
	glRasterPos2i(varr.pos[42].x + 5,varr.pos[42].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));


	str = "Hans-Peter Kriegel";
	n = strlen(str);
	glRasterPos2i(varr.pos[44].x + 5,varr.pos[44].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));

	str = "Wei Wang";
	n = strlen(str);
	glRasterPos2i(varr.pos[112].x + 5,varr.pos[112].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));

	str = "Thomas S. Huang";
	n = strlen(str);
	glRasterPos2i(varr.pos[149].x + 5,varr.pos[149].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));


	str = "HongJiang Zhang";
	n = strlen(str);
	glRasterPos2i(varr.pos[152].x + 5,varr.pos[152].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));

	str = "Yu Zhang";
	n = strlen(str);
	glRasterPos2i(varr.pos[158].x + 5,varr.pos[158].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));

	str = "Qiang Yang";
	n = strlen(str);
	glRasterPos2i(varr.pos[161].x + 5,varr.pos[161].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));


	str = "Beng Chin Ooi";
	n = strlen(str);
	glRasterPos2i(varr.pos[174].x + 5,varr.pos[174].y - 5);
	for (int i = 0; i < n; i++)
		glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *(str+i));
#endif


	glutSwapBuffers();
}


void InitGraphics(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	int window_width = 1024;
	int window_height = 1024;
	glutInitWindowSize(window_width,window_height);
	glutCreateWindow("graph draw");

	glutDisplayFunc(renderScene);
	glutIdleFunc(renderScene);
	glutReshapeFunc(changeSize);
	glutKeyboardFunc(processNormalKeys);
	glutSpecialFunc(processSpecialKeys);
	/*glutMouseFunc(mouse);
	glutMotionFunc(motion);*/


	glewInit();
	if (glewIsSupported("GL_VERSION_2_0"))
		printf("Ready for OpenGL 2.0\n");
	else {
		printf("OpenGL 2.0 not supported\n");
		exit(1);
	}

	glClearColor(0.0,0.0,0.0,1.0);
	glDisable(GL_DEPTH_TEST);


	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	
	//point smooth
	glEnable (GL_POINT_SMOOTH);
	glHint (GL_POINT_SMOOTH, GL_NICEST);

	//line smooth
	glEnable (GL_LINE_SMOOTH);
	glHint (GL_LINE_SMOOTH, GL_NICEST);

	//polygon smooth
	glEnable (GL_POLYGON_SMOOTH);
	glHint (GL_POLYGON_SMOOTH, GL_NICEST);

	glBlendFunc(GL_SRC_ALPHA, GL_DST_ALPHA);

	glDisable(GL_DEPTH_TEST);
	//Init_Shaders();
	//Set_Node_Shaders();
	






	int vertexNum = 128;
	const int maxDegree = 1;
	const int vertex_vis_t = 0;
	const int edge_vis_t = 8;
	const unsigned char vertex_alpha = 255;
	const unsigned char edge_alpha = 128;
	Medusa_Init_Config(true, EDGE_MESSAGE);


	//generate graph
	GenericReader g_reader;
	hash_map<int, int> L2P;

	g_reader.ReadGraph("I:\\Graph Dataset\\32 graph.txt",false,0,0, false, L2P);


	//vertexNum = g_reader.graph.vertexNum;


	int *set_info;


	D_VertexArray d_varr;
	D_EdgeArray d_ea;
	int *levelOffset;


	//g_reader.graph.sortByDegree();

	//no user defined constructor, must be initialized manually
	d_varr.size = 0;
	d_ea.size = 0;
	ea.buildAA(g_reader.graph);


	//set edge color
	int max_edge_weight = 0;
	for(int i = 0; i < ea.size; i ++)
		if(ea.weight[i] > max_edge_weight)
			max_edge_weight = ea.weight[i];
	printf("max_edge_weight = %d\n",max_edge_weight);
	int edge_rendered = 0;
	for(int i = 0; i < ea.size; i ++)
	{
		//(float)ea.weight[i]/(float)max_edge_weight*
		if(ea.weight[i] > edge_vis_t)
		{
			int rb;
			rb = (float)ea.weight[i]/(float)max_edge_weight*255;
			if(rb < 40)
				rb = 40;
			edge_rendered ++;
			ea.color[i*2] = make_uchar4(0, rb, 0,edge_alpha );
			ea.color[i*2 + 1] = make_uchar4(0, rb, 0,edge_alpha );
		}
		else
		{
			ea.color[i*2] = make_uchar4(0,0,0,edge_alpha );//ea.weight[i]/max_edge_weight, ea.weight[i]/max_edge_weight, ea.weight[i]/max_edge_weight);
			ea.color[i*2 + 1] = make_uchar4(0,0,0,edge_alpha );//ea.weight[i]/max_edge_weight, ea.weight[i]/max_edge_weight, ea.weight[i]/max_edge_weight);
		}
	}
	printf("edge_rendered = %d\n",edge_rendered );



	d_ea.Fill(ea);

	varr.build(g_reader.graph);

	
	

	//initialized vertex color
	
	for(int i = 0; i < varr.size; i++)
	{

		varr.color[i*4] = make_uchar4(180,180,180,vertex_alpha);
		varr.color[i*4 + 1] = make_uchar4(180,180,180,vertex_alpha);
		varr.color[i*4 + 2] = make_uchar4(180,180,180,vertex_alpha);
		varr.color[i*4 + 3] = make_uchar4(180,180,180,vertex_alpha);
	}
	

	int root_ver_pid;
	///* set the color of Jiawei Han */
	

	if(1)
	{
		printf("root weight = %d\n",vertex_weight[L2P[728850]]);
		root_ver_pid = L2P[728850];
		printf("root_ver_pid  = %d\n",root_ver_pid );
		varr.color[root_ver_pid*4] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 1] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 2] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 3] = make_uchar4(225,0,0,vertex_alpha);
	}
#ifdef HJW	
	/* set the color of Xifeng yan */
	if(1)
	{
		printf("root weight = %d\n",vertex_weight[L2P[212392]]);
		root_ver_pid = L2P[212392];
		printf("root_ver_pid  = %d\n",root_ver_pid );
		varr.color[root_ver_pid*4] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 1] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 2] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 3] = make_uchar4(225,0,0,vertex_alpha);
	}

	///* set the color of Philip S. Yu*/
	if(1)
	{
		printf("root weight = %d\n",vertex_weight[L2P[398094]]);
		root_ver_pid = L2P[398094];
		printf("root_ver_pid  = %d\n",root_ver_pid );
		varr.color[root_ver_pid*4] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 1] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 2] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 3] = make_uchar4(225,0,0,vertex_alpha);
	}
	///* set the color of Jian Pei*/
	if(1)
	{
		printf("root weight = %d\n",vertex_weight[L2P[71652]]);
		root_ver_pid = L2P[71652];
		printf("root_ver_pid  = %d\n",root_ver_pid );
		varr.color[root_ver_pid*4] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 1] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 2] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 3] = make_uchar4(225,0,0,vertex_alpha);
	}

	/* set the color of Dong Xin*/
	if(1)
	{
		printf("root weight = %d\n",vertex_weight[L2P[678292]]);
		root_ver_pid = L2P[678292];
		printf("root_ver_pid  = %d\n",root_ver_pid );
		varr.color[root_ver_pid*4] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 1] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 2] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 3] = make_uchar4(225,0,0,vertex_alpha);
	}

	/* set the color of Chin-Chen Chang */
	if(1)
	{
		printf("root weight = %d\n",vertex_weight[L2P[758974]]);
		root_ver_pid = L2P[758974];
		varr.color[root_ver_pid*4] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 1] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 2] = make_uchar4(225,0,0,vertex_alpha);
		varr.color[root_ver_pid*4 + 3] = make_uchar4(225,0,0,vertex_alpha);
	}
#endif

	/*process the marked vertexes(用于2-hop在全局图中的着色)*/
#define BFS_MARK
#ifdef BFS_MARK
	printf("marked %d vertexes\n",two_hop_logical_id.size());
	for(int i = 0; i < two_hop_logical_id.size(); i ++)
	{
		if(L2P.find(two_hop_logical_id.at(i)) != L2P.end())
		{
			int mark_ver_pid = L2P[two_hop_logical_id.at(i)];
			varr.color[mark_ver_pid*4] = make_uchar4(225,0,0,vertex_alpha);
			varr.color[mark_ver_pid*4 + 1] = make_uchar4(225,0,0,vertex_alpha);
			varr.color[mark_ver_pid*4 + 2] = make_uchar4(225,0,0,vertex_alpha);
			varr.color[mark_ver_pid*4 + 3] = make_uchar4(225,0,0,vertex_alpha);
		}
	}	
#endif
	//initialize vertex postion and speed
	//in future versions this should be inputted at the graph building phase
	//application specific data
	unsigned int range = sqrt((double)vertexNum)*20;
	unsigned int half_range = range/2;
	
	printf("max degree = %d\n",max_weight );
	int vertex_rendered = 0;

	for(int i = 0; i < varr.size; i++)
	{
		//	printf("%d %d %d",varr.color[i].x,varr.color[i].y,varr.color[i].z);
		//initialize using random coordinates
		varr.pos[i] = make_float2(float(rand32()%range),float((rand()%range)));
		varr.pos[i].x -= half_range;
		varr.pos[i].y -= half_range;
		varr.speed[i] = make_float2(0,0);
		if(vertex_weight[i] > vertex_vis_t)
		{

			//printf("vertex_weight[%d] = %d\n",i,vertex_weight[i]);
			varr.radius[i] = (float)vertex_weight[i]/(float)max_weight*10.0;
			vertex_rendered ++;
			/*if(varr.radius[i] > 10)
			{
				int temp = varr.radius[i] - 10;
				varr.radius[i] = varr.radius[i] - (varr.radius[i] - 10);
				varr.radius[i] += temp*0.15;
			}
			if(varr.radius[i] == 0)
				varr.radius[i] = 1;*/
		}
		else
		{
			varr.radius[i] = 1;
			varr.color[i*4] = make_uchar4(0,0,0,vertex_alpha);
			varr.color[i*4 + 1] = make_uchar4(0,0,0,vertex_alpha);
			varr.color[i*4 + 2] = make_uchar4(0,0,0,vertex_alpha);
			varr.color[i*4 + 3] = make_uchar4(0,0,0,vertex_alpha);
		}
		
	}

	d_varr.Fill(varr);
	printf("varr built, %d vertices rendered\n", vertex_rendered);


	//init system
	Medusa_Init_Data(ea, varr, d_ea, d_varr);

	//CPU FD_Layout Algorithm
	float t;

	cutCreateTimer(&timer);

	
	GRAPH_STORAGE_CPU::com.init(CUDPP_FLOAT, CUDPP_ADD);//must be initialized after init_Medusa
	InitInterOp();
#ifdef CPU_EXEC
	Init_CPU_Medusa(ea.size);
#endif
}

#endif
