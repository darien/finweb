import dataman.dataloader as dl; reload(dl)
import cPickle as pickle

# filename = 'r3000_tickers.csv'
# tickers = dl.loadTickers(filename)
# 
# startdt = '1991-01-01'
# enddt = '2011-01-01'

# dl.pickleBasketFromYahoo(tickers, startdt, enddt, 'marketdata/pickles/r3k_91to11/')


filename = 'nasdaq100_tickers.csv'
tickers = dl.loadTickers(filename)

startdt = '2000-01-01'
enddt = '2011-01-01'

dl.pickleBasketFromYahoo(tickers, startdt, enddt, 'marketdata/pickles/ndq100_00to11/')