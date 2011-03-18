import numpy as np
import cPickle as pickle

from time import time
from matplotlib.pyplot import figure, subplot, plot

import dataman.dataloader as dl; reload(dl)
import utilfuncs.finutils as fn; reload(fn)
import gfuncs as gf; reload(gf)

#----------------------------------------------------------
# setup
# get data
# basket, ordinals = dl.loadPickleBasket('marketdata/pickles/r3k_91to11/')
basket, ordinals = dl.loadPickleBasket('marketdata/pickles/ndq100_00to11/')
print 'data loaded. starting test...'

# params
win = 14
alpha = .2

#----------------------------------------------------------
# device execution
s = time()
dRSI = gf.gRSIBasket(basket, win, alpha)
e = time()
device_time = e-s
print 'device time: %f' % (device_time)

#----------------------------------------------------------
# serial execution
s=time()
hRSI = []
for series in basket:
	hRSI.append(fn.movingrsi(series, win, alpha))
e = time()
host_time = e-s
print 'host time: %f' % (host_time)
print 'GPU speedup: %f' % (host_time/device_time)

#----------------------------------------------------------
# plotting
fig = figure()

ax1 = fig.add_subplot(211)
ax1.plot(dRSI[0] - hRSI[0], '-b')	# check for errors
ax1.set_xlabel('')
ax1.set_ylabel('diff')
ax1.set_title('CPU/GPU calc diff')

ax2 = fig.add_subplot(212)
ax2.plot(dRSI[0], '-g')			# compare shapes
ax2.plot(hRSI[0] + .1, '-r')
ax2.set_xlabel('time')
ax2.set_ylabel('RSI')
ax2.set_title('CPU/GPU MS (offset .1)')