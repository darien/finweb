import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

from kernels import kMAcross, kRSI

def gMarketSentiment(basket, shortwin, longwin):
	
	B = np.array(basket)
	lenB = len(B)
	MS = np.zeros_like(B[0])

	for S in B:
		
		# send to device
		dS = gpuarray.to_gpu(S.astype(np.float32))
		dR = gpuarray.to_gpu(np.zeros_like(S))
		dSW = gpuarray.to_gpu(np.array(shortwin).astype(np.float32))
		dLW = gpuarray.to_gpu(np.array(longwin).astype(np.float32))

		# call function
		# this section is specific to MY gpu
		numgrids = len(S)/512 + 1
		kMAcross(dS, dR,  dSW, dLW, block=(512, 1, 1), grid=(numgrids, 1))

		# get data from device
		hR = dR.get()
		MS += hR
		
	MS /= lenB
	
	return MS
	
def gRSIBasket(basket, win, alpha):
	
	B = np.array(basket)
	bRSI = []
	
	for S in B:
		
		# send to device
		dS = gpuarray.to_gpu(S.astype(np.float32))
		dR = gpuarray.to_gpu(np.zeros_like(S))
		dW = gpuarray.to_gpu(np.array(win).astype(np.float32))
		dA = gpuarray.to_gpu(np.array(alpha).astype(np.float32))

		# call function
		# this section is specific to MY gpu
		numgrids = len(S)/512 + 1
		kRSI(dS, dR, dW, dA, block=(512, 1, 1), grid=(numgrids, 1))

		dRSI = dR.get()
		bRSI.append(dRSI)
		
	return bRSI
	
def gMarketStrength(basket, win, alpha):
	B = np.array(basket)
	lenB = len(B)
	MStren = np.zeros_like(B[0])	
	
	for S in B:
		
		# send to device
		dS = gpuarray.to_gpu(S.astype(np.float32))
		dR = gpuarray.to_gpu(np.zeros_like(S))
		dW = gpuarray.to_gpu(np.array(win).astype(np.float32))
		dA = gpuarray.to_gpu(np.array(alpha).astype(np.float32))

		# call function
		# this section is specific to MY gpu
		numgrids = len(S)/512 + 1
		kRSI(dS, dR, dW, dA, block=(512, 1, 1), grid=(numgrids, 1))

		# get data from device
		hR = dR.get()
		MStren += hR

	MStren /= lenB

	return MStren
