import urllib
import numpy as np
from scikits.timeseries import *
from scikits.timeseries.lib.interpolate import *
from datetime import datetime
import cPickle as pickle
import os, re

def loadTickers(filename):
	f = open('marketdata/csv/' + filename, 'r')
	filestring = f.read()
	tickerlist = filestring.split('\n')
	return tickerlist
	
def fetchYahooData(tickers, startdt, enddt):
	
	datearray = date_array(startdt, enddt, freq='b')
	datesTS = time_series(data=np.zeros(datearray.size), dates=datearray, freq='b', dtype=np.float32)
	
	startdt = datetime.strptime(startdt, '%Y-%m-%d')
	enddt = datetime.strptime(enddt, '%Y-%m-%d')
	
	basket = []
	for ticker in tickers:
		
		syear = str(startdt.year)
		smonth = str(startdt.month - 1)
		sday = str(startdt.day)
	
		eyear = str(enddt.year)
		emonth = str(enddt.month - 1)
		eday = str(enddt.day)
	
		url = 'http://ichart.finance.yahoo.com/table.csv?s=' + ticker + \
		'&a=' + smonth + '&b=' + sday + '&c=' + syear + \
		'&d=' + emonth + '&e=' + eday + '&f=' + eyear + \
		'&g=d&ignore=.csv'
	
		data = urllib.urlopen(url)
		linearray=data.read().split('\n')
	
		if 'Not Found' not in linearray[1]:
			dtA=[]
			adj_closeA=[]
	
			for line in reversed(linearray[1:-1]):
				temparray = line.split(',')
				dtA.append(temparray[0])
				adj_closeA.append(temparray[6])

			adjcloseTS = time_series(data=adj_closeA, dates=dtA, freq='b', dtype=np.float32)
			adjcloseTS, datesTS = align_series(adjcloseTS, datesTS, start_date=datesTS.start_date, end_date=datesTS.end_date)

			adjcloseTS = fill_missing_dates(adjcloseTS)
			adjcloseTS = forward_fill(adjcloseTS)
			adjcloseTS = backward_fill(adjcloseTS)

			basket.append(np.array(adjcloseTS.data))
		else:
			print '	data not found for ' + ticker
	
	ordinals = [datetime.fromordinal(date.toordinal()) for date in datearray]
	
	return basket, ordinals

def pickleBasketFromYahoo(tickers, startdt, enddt, filedest):
	
	existfiles = os.listdir(filedest)
	
	syear = str(datetime.strptime(startdt, '%Y-%m-%d').year)[-2:]
	eyear = str(datetime.strptime(enddt, '%Y-%m-%d').year)[-2:]
	
	idx = 0;
	ords = 0
	
	for ticker in tickers:
		filename = ticker+'_'+syear+'to'+eyear+'.p'
		exists = 0
		
		if filename not in existfiles:
			print str(idx) + ' loading: ' + ticker
			try:
				basket, ordinals = fetchYahooData([ticker], startdt, enddt)
			except:
				print '	something got fucked up when loading ' + ticker
			
			if len(basket) > 0:
				pickle.dump(basket[0], open(filedest + filename, 'wb'))
				ords += 1
		else:
			print str(idx) + ' data already loaded for ' + ticker
		
		idx += 1
	
	if 'ordinals.p' not in existfiles and ords:
		pickle.dump(ordinals, open(filedest+'ordinals.p', 'wb'))
		print 'creating ordinals file'
	else:
		print 'no ordinals file created'
		
def loadPickleBasket(filedir):

	files = os.listdir(filedir)
	pfiles = []
	for f in files:
		pf = re.findall(r'.*(?<!ordinals)\.p$', f)
		if len(pf) > 0: pfiles.append(pf[0])
		
	basket = []
	for p in pfiles:
		series = pickle.load(open(filedir+p, 'r'))
		basket.append(series)
	
	ordinals = pickle.load(open(filedir+'ordinals.p', 'r'))
	
	return basket, ordinals


