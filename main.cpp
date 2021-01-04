#include "common.h"
#include "neural.h"
#include "data.h"
#include "graphics.h"

const int nSol=3;
const int nMut=10;
const int nSamples=1;

//#define TEST
#ifdef TEST
const int imageSize=4;
const int pixelSize=8;
///const int bwRatio=3;
#else
const int imageSize=28;
const int pixelSize=2;
#endif



// data storage for custom input and output
float output[imageSize*imageSize];
float input[nSamples][imageSize*imageSize];

float nnErr(nn::NN* nn, float* data)
{
	float error=0.0;
	for(int i=0; i<nn->out->nres; i++)
		error+=abs(nn->out->res[i]-data[i]);
	return error;
}

// This function finds a random instance of a digit

int findDigit(int digit)
{
	int index;
	do
	index=rand()%nTrain;
	while(train_label[index]!=digit);
	return index;
}

// This function finds a specfic instance of a digit

int findDigitSpecific(int digit, int id)
{
	int counter=0;
	for(int i=0; i<nTrain; i++)
		if(train_label[i]==digit)
		{
			if(counter==id)
				return i;
			else
				counter++;
		}
}

struct Solution
{
	nn::NN nn;
	float error;
	Solution()
	{
		error=0.0;

#ifndef TEST
		nn.setInputLayer(28, 28);
		nn.addConvLayer(5, 5, 1);
		nn.addMaxPoolLayer();
		nn.addConvLayer(5, 5, 1);
		nn.addMaxPoolLayer();

		nn.addSigmLayer(5);
		nn.addSigmLayer(2);
		nn.addSigmLayer(5);
		nn.addSigmLayer(4*4*10);
		nn.addReshapeLayer(4, 4, 10);

		nn.addUnpoolLayer();
		nn.addTrConvLayer(5, 5, 1);
		nn.addUnpoolLayer();
		nn.addTrConvLayer(5, 5, 10);
#else
		nn.setInputLayer(28, 28);
		nn.addConvLayer(5, 5, 1);
		nn.addMaxPoolLayer();
		nn.addConvLayer(5, 5, 1);
		nn.addMaxPoolLayer();
#endif

		nn.list_params();
	}
};

int main()
{
	// initialize
	srand(time(0)*100);
	loadData();
	initGraphics();

	// create an array of pointers
	Solution* sol=new Solution[nSol];

	// test: create arbitrary input
/*
#ifdef TEST
	for(int i=0; i<nSamples; i++)
		for(int j=0; j<imageSize*imageSize; j++)
			input[i][j]=float(rand()%(bwRatio+1)/bwRatio);
#else
*/
	for(int i=0; i<nSamples; i++)
	{
		int index=findDigitSpecific(i, 0);
		normalizeToFloat(input[i], train_image[index], 784, 0.1, 0.9);
	}
//#endif
	
	// main loop
	int counter=0;
	while(true)
	{
		// this counts the generations
		counter++;
		printf("generation %i:\n", counter);

#ifndef TEST
		// calculate errors
		// and find the fittest individual
		int id=0;
		for(int i=0; i<nSol; i++)
		{
			// zero error
			sol[i].error=0.0;

			// TEST: feed forward sample
			for(int j=0; j<nSamples; j++)
			{
				sol[i].nn.ff(input[j]);

				sol[i].error+=nnErr(&sol[i].nn, input[j]);

			}

			// find the fittest individual
			if(sol[i].error<sol[id].error)
				id=i;
		}
#endif

		// TEST: visualize input and output
		for(int i=0; i<nSamples; i++)
		{
			sol[0].nn.ff(input[i]);
			for(int j=0; j<imageSize*imageSize; j++)
				output[j]=sol[0].nn.out->res[j];
#ifndef TEST
			drawNormImageBW(input[i], imageSize, imageSize, i*pixelSize*(imageSize), 0, pixelSize);
			drawNormImageBW(output, imageSize, imageSize, i*pixelSize*(imageSize), pixelSize*(imageSize), pixelSize);
#else
			drawNormImageBW(output, imageSize, imageSize, i*pixelSize*(imageSize+2), pixelSize*(imageSize), pixelSize);
#endif
		}

#ifndef TEST
		// copy parameters
		for(int i=0; i<nSol; i++)
			for(int j=0; j<sol[0].nn.nParams; j++)
				*sol[i].nn.params[j]=*sol[id].nn.params[j];

		// apply mutation
		for(int i=0; i<nMut; i++)
		{
			int mut_id=rand()%sol[0].nn.nParams;
			for(int j=1; j<nSol; j++)
				*sol[j].nn.params[mut_id]+=randf(-1.0, 1.0);
		}
#endif

		// render
		SDL_UpdateTexture(sdl_texture, 0, sdl_pixels, sdl_x*sizeof(Uint32));
		SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, sdl_texture, NULL, NULL);
        SDL_RenderPresent(sdl_renderer);
	}
}
