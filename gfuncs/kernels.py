import pycuda.autoinit
from pycuda.compiler import SourceModule

#----------------------------------------------------------
# short/long moving average crossover

kMAcross_mod = SourceModule("""
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

kMAcross = kMAcross_mod.get_function("kMAcross")

#----------------------------------------------------------
# relative strength index

kRSI_mod = SourceModule("""
	__global__ void kRSI(float *dS, float *dR, float *dW, float *dA)
	{
		int idx = blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.y+ threadIdx.x;
		
		int W = (int)dW[0];
		float A = (float)dA[0];
		const int maxW = 28;
				
		if(idx < W-1)
		{
			int i;
			float total = 0.0;
			for(i = 0; i <= idx; i++)
			{
				total = total + dS[i];
			}
			dR[idx] = (float)total/(idx+1);
		}
		else
		{
			if(W <= 1)
			{
				dR[idx] = dS[idx];
			}
			else
			{
				int i, j, k;
				float delta[maxW-1];
				float upchange[maxW-1];
				float downchange[maxW-1];
				float upema[maxW-1];
				float downema[maxW-1];
			
				// compute daily change
				for(i = 1; i <= W-1; i++)
				{
					delta[i-1] = dS[idx-W+i+1] - dS[idx-W+i];
				}
			
				// compute upchange and downchange
				for(j = 0; j < W-1; j++)
				{
					if(delta[j] >= 0)
					{
						upchange[j] = delta[j];
						downchange[j] = 0;
					} 
					else if(delta[j] < 0)
					{
						upchange[j] = 0;
						downchange[j] = abs(delta[j]);
					}
				}
			
				// compute ema of upchange and downchange
				upema[0] = upchange[0];
				downema[0] = downchange[0];
				
				for(k = 1; k < W-1; k++)
				{
					upema[k] = A * upchange[k] + (1 - A) * upema[k-1];
					downema[k] = A * downchange[k] + (1 - A) * downema[k-1];
				}
				
				dR[idx] = 100.0 - (100.0 / (1.0 + (float)upema[W-2] / downema[W-2]));
			}
		}
	}
	""")

# get the kernel
kRSI = kRSI_mod.get_function("kRSI")


