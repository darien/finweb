import numpy as np
	
def movingaverage(series, win):
	rng = np.arange(0, len(series))
	temp = [np.mean(series[:i+1]) if i<win-1 else np.mean(series[i-win+1:i+1]) for i in rng]
	return np.array(temp)
	
def marketsentiment(basket, shortwin, longwin):
	MS = np.zeros_like(basket[0])
	lenB = len(basket)
	for series in basket:
		series = np.array(series)
		beatsma = 1.0 * \
		(movingaverage(series, shortwin) > \
		movingaverage(series, longwin))
		MS += beatsma
	MS /= lenB
	
	return MS
	
def dailychange(span, direc=None):
	delta = span[1:] - span[:-1]
	if direc == 'up': return np.array([el if el >= 0 else 0 for el in delta])
	elif direc == 'down': return np.array([abs(el) if el < 0 else 0 for el in delta])
	elif direc == None: return delta
		
def ema(series, emalpha=.2):
	temp = [series[0]]
	if len(series) > 1:
		for i in np.arange(1, len(series)):
			temp.append(emalpha * series[i] + (1 - emalpha) * temp[i - 1])
		
	return np.array(temp[-1])
	
def rsindex(span, emalpha):
	span = np.array(span)
	
	upchange = dailychange(span, 'up')
	downchange = dailychange(span, 'down')

	upema = ema(upchange, emalpha)
	downema = ema(downchange, emalpha)

	rsi = 100 - (100/(1 + upema/downema))
	return rsi
		
def movingrsi(series, win=14, emalpha=.2):
	rng = np.arange(0, len(series))
	temp = [np.mean(series[:i+1]) if i<win-1 else rsindex(series[i-win+1:i+1], emalpha) \
	for i in rng]
	return np.array(temp)
	
def movingema(series, win, emalpha=.2):
	rng = np.arange(0, len(series))
	temp = [ema(series[:i+1], emalpha) if i<win-1 else ema(series[i-win+1:i+1], emalpha) \
	for i in rng]
	return np.array(temp)
		
		
	
		