import numpy as np
import cPickle as pickle

from time import time
from matplotlib.pyplot import figure, subplot, plot

import dataman.dataloader as dl; reload(dl)
import utilfuncs.finutils as fn; reload(fn)

basket, ordinals = dl.loadPickleBasket('marketdata/pickles/r3k_91to11/')

# ### GPU Stuff ###
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


# RSI params
series = basket[0]
win = 14
alpha = .2

# device test
s = time()
mod = SourceModule("""
	__global__ void kRSI(float *dS, float *dR, float *dW, float *dA)
	{
		int idx = blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.y+ threadIdx.x;
		
		int W = (int)dW[0];
		float A = (float)dA[0];
		const int maxW = 28;
				
		if(idx < W-1)
		{
			int i;
			float total = 0.0;
			for(i = 0; i <= idx; i++)
			{
				total = total + dS[i];
			}
			dR[idx] = (float)total/(idx+1);
		}
		else
		{
			if(W <= 1)
			{
				dR[idx] = dS[idx];
			}
			else
			{
				int i, j, k;
				float delta[maxW-1];
				float upchange[maxW-1];
				float downchange[maxW-1];
				float upema[maxW-1];
				float downema[maxW-1];
			
				// compute daily change
				for(i = 1; i <= W-1; i++)
				{
					delta[i-1] = dS[idx-W+i+1] - dS[idx-W+i];
				}
			
				// compute upchange and downchange
				for(j = 0; j < W-1; j++)
				{
					if(delta[j] >= 0)
					{
						upchange[j] = delta[j];
						downchange[j] = 0;
					} 
					else if(delta[j] < 0)
					{
						upchange[j] = 0;
						downchange[j] = abs(delta[j]);
					}
				}
			
				// compute ema of upchange and downchange
				upema[0] = upchange[0];
				downema[0] = downchange[0];
				
				for(k = 1; k < W-1; k++)
				{
					upema[k] = A * upchange[k] + (1 - A) * upema[k-1];
					downema[k] = A * downchange[k] + (1 - A) * downema[k-1];
				}
				
				dR[idx] = 100.0 - (100.0 / (1.0 + (float)upema[W-2] / downema[W-2]));
			}
		}
	}
	""")

# get the kernel
kRSI = mod.get_function("kRSI")

# host data
S = series.astype(np.float32)
R = np.zeros_like(S)
W = np.array(win).astype(np.float32)
A = np.array(alpha).astype(np.float32)

dS = gpuarray.to_gpu(S)
dR = gpuarray.to_gpu(R)
dW = gpuarray.to_gpu(W)
dA = gpuarray.to_gpu(A)

# call function
numgrids = len(S)/512 + 1
kRSI(dS, dR, dW, dA, block=(512, 1, 1), grid=(numgrids, 1))

dRSI = dR.get()

e = time()
print 'device time: %f' % (e-s)

# serial execution
s=time()
hRSI = fn.movingrsi(series, win, alpha)
e = time()
host_time = e-s
print 'host time: %f' % (host_time)
print 'GPU speedup: %f' % (host_time/device_time)

#.......................................................
# plotting
fig = figure()

ax1 = fig.add_subplot(211)
ax1.plot(dRSI - hRSI, '-b')	# check for errors
ax1.set_xlabel('')
ax1.set_ylabel('diff')
ax1.set_title('CPU/GPU calc diff')

ax2 = fig.add_subplot(212)
ax2.plot(dRSI, '-g')			# compare shapes
ax2.plot(hRSI + .1, '-r')
ax2.set_xlabel('time')
ax2.set_ylabel('RSI')
ax2.set_title('CPU/GPU MS (offset .1)')
