# pycuda basics:
# 1) create/execute kernel
# 2) maniuplate gpu_array object
# 3) elementwise kernel
# 4) gpu map/reduce kernel
# 5) compare elementwise/sum with mapreduce
# 6) test using threadIdx in computations

###################
# 1) kernels

print '\npycuda kernel experiment 1'
print '--------------------------\n'

import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# define CUDA kernel like this
mod = SourceModule("""
  __global__ void doublify(float *a, float *a2)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a2[idx] = 2*a[idx];
  }
  """)

# get the kernel as a function
func = mod.get_function("doublify")

# create some 32-bit data
a = numpy.random.randn(4,4)
a = a.astype(numpy.float32)

# create place to put result
a2 = numpy.zeros([4,4])
a2 = a2.astype(numpy.float32)

# allocate device memory
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

a2_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a2_gpu, a2)

# call function with input data and block structure
func(a_gpu, a2_gpu, block=(4,4,1))

# allocate host memory for result and copy result from device
a_doubled = numpy.empty_like(a2)
cuda.memcpy_dtoh(a_doubled, a2_gpu)

print a
print '\n'
print a_doubled

# alternate kernel call using cuda driver in/out

import pycuda.driver as drv

func(drv.In(a), drv.Out(a2), block=(4,4,1))

print a
print '\n'
print a2

###################
# 2) gpu arrays
# long gpu_array expressions create many intermediate variables
# use elementwise kernel instead

print '\npycuda gpu_array experiment 1'
print '-----------------------------\n'

import pycuda.gpuarray as gpuarray

a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()

print "original array:"
print a_gpu
print "doubled with gpuarray:"
print a_doubled

# something more complex
print '\nmore complex:\n'
from time import time

## host execution
# allocate host arrays
sz = 1000
hA = numpy.random.randn(sz, sz).astype(numpy.float32)
hB = numpy.random.randn(sz, sz).astype(numpy.float32)
hC = numpy.zeros([sz, sz]).astype(numpy.float32)

# time operation
s = time()
# hC = hA * hB
hC = sum(hA)
e = time()

print 'serial elapsed time: %f \n' % (e-s)

## device execution
# allocate device arrays
dA = gpuarray.to_gpu(hA)
dB = gpuarray.to_gpu(hB)
dC = gpuarray.to_gpu(hC)

# time operation
s = time()
# dC = dA * dB
dC = gpuarray.sum(dA)
e = time()

print 'gpu elapsed time: %f \n' % (e-s)

# cumath functions
print '\ncumath stuff:\n'

from pycuda import cumath

# time operation
s = time()
hC = numpy.log(hA)
e = time()

print 'serial elapsed time: %f \n' % (e-s)

## device execution
# allocate device arrays
dA = gpuarray.to_gpu(hA)
dB = gpuarray.to_gpu(hB)
dC = gpuarray.to_gpu(hC)

# time operation
s = time()
dC = cumath.log(dA)
e = time()

print 'gpu elapsed time: %f \n' % (e-s)

###################
# 3) elementwise kernel
# performs array operations much faster than gpu_array

print '\n elementwise kernel\n'
print '---------------------\n'

from pycuda.curandom import rand as curand

a_gpu = curand((1000,))
b_gpu = curand((1000,))

from pycuda.elementwise import ElementwiseKernel
lin_comb = ElementwiseKernel(
        "float a, float *x, float b, float *y, float *z",
        "z[i] = a*x[i] + b*y[i]",
        "linear_combination")

c_gpu = gpuarray.empty_like(a_gpu)

s = time()
lin_comb(5, a_gpu, 6, b_gpu, c_gpu)
e = time()
print 'elementwise kernel elapsed time: %f \n' % (e-s)

s = time()
c = (5*a_gpu+6*b_gpu).get()
e = time()
print 'gpu_array elapsed time: %f \n' % (e-s)

a = numpy.random.randn(1000,1000)
b = numpy.random.randn(1000,1000)
c = numpy.zeros([1000,1000])

s = time()
c = 5*a+6*b
e = time()
print 'cpu elapsed time: %f \n' % (e-s)

###################
# 4) map/reduce kernel

print '\n map/reduce kernel\n'
print '--------------------\n'

from pycuda.reduction import ReductionKernel

sz = 7000

# on device
a = gpuarray.arange(sz, dtype=numpy.float32)
b = gpuarray.arange(sz, dtype=numpy.float32)

krnl = ReductionKernel(numpy.float32, neutral="0",
        reduce_expr="a+2*b+b*b", map_expr="x[i] + y[i]",
        arguments="float *x, float *y")

# device perf
s = time()
my_dot_prod = krnl(a, b).get()
e = time()
print 'kernel time: %f' % (e-s)

# on host
a2 = arange(sz, dtype=numpy.float32)
b2 = arange(sz, dtype=numpy.float32)

# host perf
s = time()
c = a2*b2
c2 = sum([a+2*b+b*b for a,b in zip(c[:-1], c[1:])])
e = time()

print 'host time: %f' % (e-s)

###################
# 5) compare elemntwise with gpuarray.sum to mapreduce
# mapreduce always wins

print '\n elementwise/sum versus mapreduce\n'
print '-----------------------------------\n'

# mapreduce
a = gpuarray.arange(sz, dtype=numpy.float32)
b = gpuarray.arange(sz, dtype=numpy.float32)

mrkrnl = ReductionKernel(numpy.float32, neutral="0",
        reduce_expr="a+b", map_expr="x[i] + y[i]",
        arguments="float *x, float *y")

s = time()
my_dot_prod = mrkrnl(a, b)
e = time()

print my_dot_prod
print 'reduction kernel time: %f' % (e-s)

# elementwise/sum
a2 = gpuarray.arange(sz, dtype=numpy.float32)
b2 = gpuarray.arange(sz, dtype=numpy.float32)

lin_comb = ElementwiseKernel(
        "float *x, float *y, float* z",
        "z[i] = x[i] + y[i]",
        "linear_combination")

c2 = gpuarray.empty_like(a2)

s = time()
lin_comb(a2, b2, c2)
cs = gpuarray.sum(c2)
e = time()

print cs
print 'elementwise kernel elapsed time: %f \n' % (e-s)

###################
# 6) experiments with custom kernel
# findings:
# maximum threads (for this card) = 512

# new kernel
mod = SourceModule("""
  __global__ void kernelSanders(float *a, float *b, float *c)
  {
	int idx = blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.y+ threadIdx.x;
	c[idx] = a[idx] * b[idx];
  }
  """)

# get the kernel
func = mod.get_function("kernelSanders")

# make some data
sz = 5000
a3 = gpuarray.arange(sz, dtype=numpy.float32)
b3 = gpuarray.arange(sz, dtype=numpy.float32)
c3 = gpuarray.empty_like(a3)

# call function
func(a3, b3, c3, block=(16, 16, 1), grid=(20, 1) )

# copy to host
cH = c3.get()







