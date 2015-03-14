#ifndef GRAPHREADER_H
#define GRAPHREADER_H
#include <string>
#include <list>

#ifdef _WIN32
	#include <hash_map>
	#define HASH_SPACE stdext
#endif

#ifdef __linux__
	#include <ext/hash_map>
	#define HASH_SPACE __gnu_cxx
#endif

#include "../MedusaRT/GraphConverter.h"
#include "../Algorithm/Configuration.h"

#ifndef TEST_DS
using namespace std;


extern vector<int> two_hop_logical_id;
extern int *vertex_weight;
extern int max_weight;
extern HASH_SPACE::hash_map<int ,int> global_L2P;


/**		The base class of graph reader
*
*	
*/
class GraphReader
{
public:
	GraphIR graph;
	GraphReader();
	/**
	*
	*
	* @param  file_name - the name of the graph file
	* @return	
	* @note	
	*
	*/
private:
	virtual void ReadGraph(string file_name);
};


/** Unweighted and undirected graph reader
*	File Format
Number_Of_Vertices Number_Of_Edges
src	dst
src	dst
...
...
src	dst

@note: 文件中不会重复一条边以表示无向，如果定点1和定点2之间有一条边，那么文件中有(1,2)则不会有(2,1)，反之亦然
*/
class UWUDReader:public GraphReader
{
	public:
	void ReadGraph(string file_name);
};


/** Unweighted and directed graph reader
*	File Format
Number_Of_Vertices Number_Of_Edges
src	dst
src	dst
...
...
src	dst
*/
class UWDReader:public GraphReader
{
	public:
	void ReadGraph(string file_name);
};




/**		DBLP graph reader
*
*/
class DBLPReader:public GraphReader
{
	public:
	void ReadGraph(string file_name);
};


/**		Patent graph reader
*
*/
class PatentReader:public GraphReader
{
	public:
	void ReadGraph(string file_name);
};

/**
*
*/
class GenericReader:public GraphReader
{
	public:
		void ReadGraph(string file_name, bool directed, int degree_threashold, int weight_threshold, bool sorted,HASH_SPACE::hash_map<int ,int> &L2P);
		
};

#endif

#endif
