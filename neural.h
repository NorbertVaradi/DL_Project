#pragma once

#include "common.h"

namespace nn
{
	struct Layer
	{
		Layer* prev;
		Layer* next;
		int type;
		// 0: input
		// 1: convolution
		// 2: maxpool
		// 3: sigmoid
		// 4: reshape
		// 5: transposed conv.
		// 6: unpool


		// Other settings
		int x;
		int y;
		int ch;
		int kx;
		int ky;

		// output
		int nres;
		float* res;
		int nwts;
		float* wts;
		float* biases;

		Layer()
		{
			prev=0;
			next=0;
			nres=0;
			res=0;
			nwts=0;
			wts=0;
			biases=0;
		}
		~Layer()
		{
			if(next)
				delete next;
			if(res)
				delete[] res;
			if(wts)
				delete[] wts;
			if(biases)
				delete[] biases;
		}

		void ff()
		{
			// perform action
			switch(type)
			{
			case 1:
				for(int i=0; i<prev->ch; i++)
					for(int j=0; j<ch/prev->ch; j++)
						convolve(res+(x*y)*(i*ch/prev->ch+j),
						prev->res+prev->x*prev->y*i,
						wts+kx*ky*j,
						prev->x,
						prev->y,
						kx,
						ky);
				break;
			case 2:
				for(int i=0; i<ch; i++)
					maxpool(res+x*y*i,
					prev->res+prev->x*prev->y*i,
					prev->x,
					prev->y);
				break;
			case 3:
				for(int i=0; i<nres; i++)
				{
					res[i]=0.0;
					for(int j=0; j<prev->nres; j++)
						res[i]+=prev->res[j]*wts[i*prev->nres+j];
					res[i]+=biases[i];
					res[i]=1.0/(1.0+exp(-res[i]));
				}
				break;
			case 4:
				for(int i=0; i<nres; i++)
					res[i]=prev->res[i];
				break;
			case 5:
				for(int i=0; i<ch; i++)
				{
					// zero output
					for(int j=0; j<x*y; j++)
						res[i*x*y+j]=0.0;

					// apply kernels
					for(int j=0; j<prev->ch/ch; j++)
						trconv(res+x*y*i,
						prev->res+(prev->x*prev->y)*(i*prev->ch/ch+j),
						wts+kx*ky*j,
						prev->x,
						prev->y,
						kx,
						ky);

					// apply sigmoid
					for(int j=0; j<x*y; j++)
						res[i*x*y+j]=1.0/(1.0+exp(-res[i*x*y+j]));

					// or apply relu?
					/*
					for(int j=0; j<x*y; j++)
					{
						if(res[i*x*y+j]<0.0)
							res[i*x*y+j]=0.0;
					}
					*/
					
				}
				break;
			case 6:
				for(int i=0; i<ch; i++)
					unpool(res+x*y*i,
					prev->res+prev->x*prev->y*i,
					prev->x,
					prev->y);
				break;
			}
			// next
			if(next)
				next->ff();
		}
	};

	struct NN
	{
		Layer* in;
		Layer* out;
		int nParams;
		float** params;

		NN()
		{
			in=0;
			params=0;
		}

		~NN()
		{
			delete in;
			if(params)
				delete[] params;
		}

		void addLayer(int _type)
		{
			// add new node to the linked list
			if(!in)
			{
				in=new Layer();
				out=in;
			}
			else
			{
				Layer* tmp=out;
				out->next=new Layer;
				out=out->next;
				out->prev=tmp;
			}

			// save arguments
			out->type=_type;
		}

		void setInputLayer(int _x, int _y)
		{
			if(!in)
			{
				addLayer(0);
				out->x=_x;
				out->y=_y;
				out->ch=1;
				out->nres=_x*_y;
				out->res=new float[out->nres];
			}
		}

		void addConvLayer(int _kx, int _ky, int _ch_out)
		{
			if(in)
			{
				addLayer(1);
				int x=out->prev->x-_kx+1;
				int y=out->prev->y-_ky+1;
				int ch=out->prev->ch*_ch_out;
				out->x=x;
				out->y=y;
				out->ch=ch;
				out->nres=x*y*ch;
				out->res=new float[out->nres];
				out->nwts=_kx*_ky*_ch_out;
				out->wts=new float[out->nwts];
				out->kx=_kx;
				out->ky=_ky;

				// randomize
				for(int i=0; i<out->nwts; i++)
					out->wts[i]=randf(-1.0, 1.0);
			}
		}

		void addMaxPoolLayer()
		{
			if(in)
			{
				addLayer(2);
				out->x=out->prev->x/2;
				out->y=out->prev->y/2;
				out->ch=out->prev->ch;
				out->nres=out->x*out->y*out->ch;
				out->res=new float[out->nres];
			}
		}
		
		void addSigmLayer(int _neurons)
		{
			if(in)
			{
				addLayer(3);
				out->nres=_neurons;
				out->res=new float[out->nres];
				out->nwts=out->prev->nres*out->nres;
				out->wts=new float[out->nwts];
				out->biases=new float[out->nres];

				// randomize
				for(int i=0; i<out->nwts; i++)
					out->wts[i]=randf(-1.0, 1.0);
				for(int i=0; i<out->nres; i++)
					out->biases[i]=randf(-1.0, 1.0);
			}
		}

		void addReshapeLayer(int _x, int _y, int _ch)
		{
			if(in)
			{
				addLayer(4);
				out->x=_x;
				out->y=_y;
				out->ch=_ch;
				out->nres=out->x*out->y*out->ch;
				out->res=new float[out->nres];
			}
		}

		void addTrConvLayer(int _kx, int _ky, int _ch_in)
		{
			if(in)
			{
				addLayer(5);
				int x=out->prev->x+_kx-1;
				int y=out->prev->y+_ky-1;
				int ch=out->prev->ch/_ch_in;
				out->x=x;
				out->y=y;
				out->ch=ch;
				out->nres=x*y*ch;
				out->res=new float[out->nres];
				out->nwts=_kx*_ky*_ch_in;
				out->wts=new float[out->nwts];
				out->kx=_kx;
				out->ky=_ky;

				// randomize
				for(int i=0; i<out->nwts; i++)
					out->wts[i]=randf(-1.0, 1.0);
			}
		}

		void addUnpoolLayer()
		{
			if(in)
			{
				addLayer(6);
				out->x=out->prev->x*2;
				out->y=out->prev->y*2;
				out->ch=out->prev->ch;
				out->nres=out->x*out->y*out->ch;
				out->res=new float[out->nres];
			}
		}

		void ff(float* data, float noise=0.0)
		{
			for(int i=0; i<in->nres; i++)
				in->res[i]=data[i]+randf(-noise, noise);
			if(in->next)
				in->next->ff();
		}

		void list_params()
		{
			if(!params)
			{
				// count model parameters
				// and allocate memory
				nParams=0;
				Layer* layer=in;
				while(layer->next)
				{
					layer=layer->next;
					nParams+=layer->nwts;
					if(layer->type==3)
						nParams+=layer->nres;
				}
				params=new float*[nParams];

				// make a list of pointers
				int index=0;
				layer=in;
				while(layer->next)
				{
					// add weights
					layer=layer->next;
					for(int i=0; i<layer->nwts; i++)
						params[index++]=&layer->wts[i];
				
					// add biases if sigmoid
					if(layer->type==3)
						for(int i=0; i<layer->nres; i++)
							params[index++]=&layer->biases[i];
				}
			}
		}
	};
}