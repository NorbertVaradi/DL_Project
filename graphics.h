#pragma once

// ********** SDL renderer ********** 

// load sdl
#define SDL_MAIN_HANDLED
#include <sdl/SDL.h>
#pragma comment(lib, "sdl2.lib")
#pragma comment(lib, "sdl2main.lib")

const int sdl_x=560;
const int sdl_y=560;
SDL_Event sdl_event;
Uint8 sdl_pixels[sdl_y][sdl_x][4];

SDL_Window* sdl_window;
SDL_Renderer* sdl_renderer;
SDL_Texture* sdl_texture;

void initGraphics()
{
	// init sdl
	SDL_Init(SDL_INIT_VIDEO);

	// create window
	sdl_window=SDL_CreateWindow("SDL2 Pixel Drawing",
        SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		sdl_x,
		sdl_y,
		0);
    sdl_renderer=SDL_CreateRenderer(sdl_window, -1, 0);
    sdl_texture=SDL_CreateTexture(sdl_renderer,
        SDL_PIXELFORMAT_ARGB8888,
		SDL_TEXTUREACCESS_STATIC,
		sdl_x,
		sdl_y);
}

void destroyGraphics()
{
	SDL_DestroyTexture(sdl_texture);
    SDL_DestroyRenderer(sdl_renderer);
	SDL_DestroyWindow(sdl_window);
    SDL_Quit();
}

void normalizeToBytes(Uint8* dst, float* src, int size)
{
	float min=src[0];
	float max=src[0];
	for(int i=0; i<size; i++)
	{
		if(src[i]<min) min=src[i];
		if(src[i]>max) max=src[i];
	}
	for(int i=0; i<size; i++)
		dst[i]=Uint8((src[i]-min)/(max-min)*255.0);
}

void normalizeToFloat(float* dst, Uint8* src, int size, float min, float max)
{
	for(int i=0; i<size; i++)
		dst[i]=((float)src[i])/255.0*(max-min)+min;
}

void drawImageBW(Uint8* data, int width, int height, int x, int y, int size=1)
{
	if(size==1)
	{
		for(int i=0; i<width; i++)
			for(int j=0; j<height; j++)
			{
				Uint8* pixel=&sdl_pixels[j+y][i+x][0];
				pixel[0]=pixel[1]=pixel[2]=data[j*width+i];
			}
		}
	else
	{
		for(int i=0; i<width; i++)
			for(int j=0; j<height; j++)
			{
				for(int k=0; k<size; k++)
					for(int l=0; l<size; l++)
					{
						Uint8* pixel=&sdl_pixels[j*size+l+y][i*size+k+x][0];
						pixel[0]=pixel[1]=pixel[2]=data[j*width+i];
					}
				
			}
		}
}

// this one is slow
void drawNormImageBW(float* data, int width, int height, int x, int y, int size=1)
{
	Uint8* tmp=new Uint8[width*height];
	normalizeToBytes(tmp, data, width*height);
	drawImageBW(tmp, width, height, x, y, size);
	delete[] tmp;
}