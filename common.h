#pragma once

#include <iostream>
#include <time.h>
using namespace std;

float randf(float min, float max)
{
	return min+((float)rand()/(float)RAND_MAX)*(max-min);
}

float norm(float min, float max, float degree)
{
	return min+(max-min)*degree;
}

float min(float a, float b)
{
	return a<b?a:b;
}

float max(float a, float b)
{
	return a>b?a:b;
}

void convolve(float* target, float* source, float* kernel, int source_x, int source_y, int kx, int ky)
{
	int target_x=source_x-kx+1;
	int target_y=source_y-ky+1;
	for(int i=0; i<target_y; i++)
	{
		for(int j=0; j<target_x; j++)
		{
			float* dest=&target[i*target_x+j];
			*dest=0.0;
			for(int k=0; k<ky; k++)
				for(int l=0; l<kx; l++)
					*dest+=source[(i*source_x+j)+(k*source_x+l)]*kernel[k*kx+l];

			// apply relu
			if(*dest<0.0)
				*dest=0.0;
		}
	}
}

// note: zero target before tr. conv. and apply relu after the process
void trconv(float* target, float* source, float* kernel, int source_x, int source_y, int kx, int ky)
{
	int target_x=source_x+kx-1;
	int target_y=source_y+ky-1;
	for(int i=0; i<source_y; i++)
		for(int j=0; j<source_x; j++)
			for(int k=0; k<ky; k++)
				for(int l=0; l<kx; l++)
					target[(i*target_x+j)+(k*target_x+l)]+=source[i*source_x+j]*kernel[k*kx+l];
}

void maxpool(float* target, float* source, int source_x, int source_y)
{
	int target_x=source_x/2;
	int target_y=source_y/2;
	for(int k=0; k<target_y; k++)
		for(int l=0; l<target_x; l++)
		{
			target[k*target_x+l]
			=max(max(source[2*k*source_x+2*l], source[2*k*source_x+2*l+1]),
				max(source[(2*k+1)*source_x+2*l], source[(2*k+1)*source_x+2*l+1]));
		}
}

void unpool(float* target, float* source, int source_x, int source_y)
{
	int target_x=source_x*2;
	int target_y=source_y*2;
	for(int k=0; k<source_y; k++)
		for(int l=0; l<source_x; l++)
			target[2*k*target_x+2*l]
			=target[2*k*target_x+2*l+1]
			=target[(2*k+1)*target_x+2*l]
			=target[(2*k+1)*target_x+2*l+1]
			=source[k*source_x+l];
}