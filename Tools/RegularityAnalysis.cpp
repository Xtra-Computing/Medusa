#include "../Tools/RegularityAnalysis.h"
#include <string>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstring>
using namespace std;

void AnalyzeRegularity(char *gt_file_name)
{
	ifstream gt_file(gt_file_name);
	if(!gt_file.good())
	{
		printf("open GT Graph File failed\n");
		exit(-1);
	}

	int v_num, e_num;
	char line[1024];
	char first_ch;
	while(gt_file.get(first_ch))
	{
		if(first_ch == 'p')
		{
			string temp;
			gt_file>>temp>>v_num>>e_num;

			gt_file.getline(line, 1024);//eat the line break
			break;
		}
		gt_file.getline(line, 1024);//eat the line break
	}
	
	int *vertex_degree = (int*) malloc(sizeof(int)*(v_num+1));
	memset(vertex_degree, 0, sizeof(int)*(v_num+1));
	

	int src_id, dst_id;
	float edge_weight;

	while(gt_file.get(first_ch))
	{
		if(first_ch == 'a')
		{
			gt_file>>src_id>>dst_id>>edge_weight;
			vertex_degree[src_id] ++;
		}
	}

	int max_degree = vertex_degree[1];
	int min_degree = vertex_degree[1];
	double total_degree = 0;
	for(int i = 1; i < v_num + 1; i ++)
	{
		total_degree += vertex_degree[i];
		if(vertex_degree[i] > max_degree)
			max_degree = vertex_degree[i];
		if(vertex_degree[i] < min_degree)
			min_degree = vertex_degree[i];
	}
	
	double average_degree = total_degree/v_num;

	//calculate standard deviation
	double standard_deviation = 0.0;
	for(int i = 1; i < v_num + 1; i ++)
	{
		standard_deviation = standard_deviation + (vertex_degree[i] - average_degree)*(vertex_degree[i] - average_degree);
	}

	standard_deviation /= v_num;
	standard_deviation = sqrt(standard_deviation);

	printf("---- Input Graph Statistics ----\n");
	printf("Max degree: %d\n", max_degree);
	printf("Min degree: %d\n", min_degree);
	printf("Average degree: %f\n", average_degree);
	printf("Standard deviation: %f\n", standard_deviation);


}
