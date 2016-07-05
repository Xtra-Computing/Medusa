#ifndef MSGCOMBINER_H
#define MSGCOMBINER_H
#include "Configuration.h"
#include "../MultipleGPU/MultiGraphStorage.h"

#ifdef __LINUX__
#include <sys/time.h>
#include <unistd.h>
#endif

#include <time.h>


//in millisecond
void printTimestamp()
{

#ifdef __LINUX__
	struct timeval  tv;
	struct timezone tz;


	struct tm      *tm;
	long long         start;

	gettimeofday(&tv, &tz);

	start = tv.tv_sec * 1000000 + tv.tv_usec;

	printf("%lld\n",start);
#endif


}

void *combine_reuse(void *did)
{
	return 0;
}

void *combine(void *did)
{
	//BFS needs no combiner
	return 0;

}

#endif
