#!/usr/bin/env bash

if hash nvidia-smi 2>/dev/null; then
  echo "building with GPU support"
  export HAS_GPU=true
else 
  echo "building without GPU support"
  export HAS_GPU=false
fi

CUDA_PATH=/n/regal/rush_lab/sw/usr/local/cuda-7.5/

cd src/

# clean everything from before
rm -f *.o *.so
rm -f internals_s.c internals_s.h

if [ "$HAS_GPU" = true ] ; then 
  echo "Compiling functions using nvcc..."
  # force compilation in CUDA/C++ mode
  nvcc -c -dc --shared functions_cuda_kernel.cu -x cu -arch=sm_35 -Xcompiler -fPIC -lcudadevrt -lcudart -o functions_cuda_kernel.cu.o -D __BOTH__='__device__ __host__'
  nvcc -c -dc --shared internals.c -x cu -arch=sm_35 -Xcompiler -fPIC -lcudadevrt -lcudart -o internals.cu.o -D __BOTH__='__device__ __host__' -include cfloat
  echo "Compiled, now linking..."

  # required intermediate device code link step
  nvcc -arch=sm_35 -dlink functions_cuda_kernel.cu.o internals.cu.o -o functions.link.cu.o -Xcompiler -fPIC -lcudadevrt -lcudart
fi

echo "Generating sanitized versions of internals for C compilation..."
echo "#include <float.h>" | cat - internals.c | sed "s/__BOTH__//g" | sed "s/internals.h/internals_s.h/g" > internals_s.c
sed "s/__BOTH__//" internals.h > internals_s.h

cd ../

echo "Building python interface to CUDA code"
if [ "$HAS_GPU" = true ] ; then 
  python3 build.py --cuda
else
  python3 build.py
fi
