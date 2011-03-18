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


# params


# device test
s = time()
mod = SourceModule("""
	__global__ void kRSI(float *dS, float *dR, float *dW, float *dA)
	{
		
	}
	""")

# get the kernel
 = mod.get_function("")

# host data

# call function
numgrids = len(S)/512 + 1
kRSI(dS, dR, dW, dA, block=(512, 1, 1), grid=(numgrids, 1))

dRSI = dR.get()

e = time()
print 'device time: %f' % (e-s)

# serial execution
s=time()

# serial code

e = time()
host_time = e-s

print 'host time: %f' % (host_time)
print 'GPU speedup: %f' % (host_time/device_time)

#.......................................................
# plotting
fig = figure()

ax1 = fig.add_subplot(211)
ax1.plot(@ - @, '-b')	# check for errors
ax1.set_xlabel('')
ax1.set_ylabel('diff')
ax1.set_title('CPU/GPU calc diff')

ax2 = fig.add_subplot(212)
ax2.plot(@, '-g')			# compare shapes
ax2.plot(@ + .1, '-r')
ax2.set_xlabel('time')
ax2.set_ylabel('RSI')
ax2.set_title('CPU/GPU MS (offset .1)')
