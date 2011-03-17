import dataman.dataloader as dl; reload(dl)
import utilfuncs.finutils as fn; reload(fn)
import cPickle as pickle
from matplotlib.pyplot import plot

basket, ordinals = dl.loadPickleBasket('marketdata/pickles/ndq100_00to11/')
series = basket[0]

win = 14

a = fn.movingrsi(series)
plot(a)
