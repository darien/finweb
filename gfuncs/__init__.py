import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

from kernels import kMAcross

def gMarketSentiment(basket, shortwin, longwin):

	# host data
	B = np.array(basket)
	lenB = len(B)
	MS = np.zeros_like(B[0])

	for S in B:
		
		S = S.astype(np.float32)
		R = np.zeros_like(S)
		SW = np.array(shortwin).astype(np.float32)
		LW = np.array(longwin).astype(np.float32)

		# send to device
		dS = gpuarray.to_gpu(S)
		dR = gpuarray.to_gpu(R)
		dSW = gpuarray.to_gpu(SW)
		dLW = gpuarray.to_gpu(LW)

		# call function
		# this section is specific to MY gpu
		numgrids = len(S)/512 + 1
		kMAcross(dS, dR,  dSW, dLW, block=(512, 1, 1), grid=(numgrids, 1))

		hR = dR.get()
		MS += hR
		
	MS /= lenB
	
	return MS
