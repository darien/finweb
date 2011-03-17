import pycuda.autoinit
from pycuda.compiler import SourceModule


# short/long moving average crossover
mod = SourceModule("""
	__global__ void kMAcross(float *dS, float *dR, float *dSW, float *dLW)
	{
		int idx = blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.y+ threadIdx.x;
		float SW = (float)dSW[0];
		float LW = (float)dLW[0];
		
		if(idx >= LW)
		{
			int i, j;
			float ss = 0, sma = 0, ls = 0, lma = 0;
		
			for(i = 0; i < SW; i++)
			{
				ss = ss + dS[idx - i];
			}
			sma = ss / SW;
			
			for(j = 0; j < LW; j++)
			{
				ls = ls + dS[idx - j];
			}
			lma = ls / LW;
			
			dR[idx] = sma > lma;
		}
	}
	""")

kMAcross = mod.get_function("kMAcross")