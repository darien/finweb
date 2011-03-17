import numpy as np
import cPickle as pickle

from time import time
from matplotlib.pyplot import figure, subplot, plot

import dataman.dataloader as dl; reload(dl)
import utilfuncs.finutils as fn; reload(fn)
import gfuncs as gf; reload(gf)

#.......................................................
# setup
# get data
basket, ordinals = dl.loadPickleBasket('marketdata/pickles/r3k_91to11/')
# basket, ordinals = dl.loadPickleBasket('marketdata/pickles/ndq100_00to11/')
print 'data loaded. starting test...'

# params
sw = 5
lw = 75

#.......................................................
# device execution
s = time()
dMS = gf.gMarketSentiment(basket, sw, lw)
e = time()
device_time = e-s
print 'device time: %f' % (device_time)

#.......................................................
# serial execution
s=time()
hMS = fn.marketsentiment(basket, sw, lw)
e = time()
host_time = e-s
print 'host time: %f' % (host_time)
print 'GPU speedup: %f' % (host_time/device_time)

#.......................................................
# plotting
fig = figure()

ax1 = fig.add_subplot(211)
ax1.plot(dMS - hMS, '-b')	# check for errors
ax1.set_xlabel('')
ax1.set_ylabel('diff')
ax1.set_title('CPU/GPU calc diff')

ax2 = fig.add_subplot(212)
ax2.plot(dMS, '-g')			# compare shapes
ax2.plot(hMS + .1, '-r')
ax2.set_xlabel('time')
ax2.set_ylabel('MS')
ax2.set_title('CPU/GPU MS (offset .1)')

