import numpy as np
from scikits.timeseries import *
from scikits.timeseries.lib.moving_funcs import mov_mean, mov_average_expw
import utilfuncs.finutils as fn; reload(fn)
from matplotlib.pyplot import plot

data = np.arange(0, 100)
dates = np.arange(0, 100)
win = 20

### compare movavg calc to timeseries movmean:
# ts mov_mean
t = time_series(data=data, dates=dates)
tMM = mov_mean(t, win)

# finutils movingaverage (with leading zeros, like timeseries)
fMA = np.array([0 if i < win-1 else np.mean(data[i-win+1:i+1]) for i in dates])

print 'moving average errors: %f' % (sum(tMM!=fMA))

### compare exp movavg calc to timeseries expw movavg
# ts expw
tEXP = mov_average_expw(t, win)

# finutils
fEXP = fn.movingema(t, win, 0.0952381)

print 'exp moving average errors: %f' % (sum(tEXP!=fEXP))

plot(tEXP-fEXP, '-g')