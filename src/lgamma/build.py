import os
import torch
from torch.utils.ffi import create_extension
import pdb
import argparse

parser = argparse.ArgumentParser(description='GPU build script')
parser.add_argument('--cuda-path', type=str, default='/usr/local/cuda-7.5/targets/x86_64-linux/lib/', help='Path to CUDA binaries (try `locate libcudart.so`)')
parser.add_argument('--path', type=str, default='')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()

sources = ['src/functions.c', 'src/internals_s.c']
headers = ['src/functions.h', 'src/internals_s.h']
defines = []
extra_objects = []
libraries = []
this_file = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available() and args.cuda:
    print('Including CUDA code.')
    sources += ['src/functions_cuda.c']
    headers += ['src/functions_cuda.h']
    defines.append(('WITH_CUDA', None))
    libraries += ["cudart", "cudadevrt"]
    extra_objects = ['src/functions.link.cu.o', 'src/internals.cu.o', 'src/functions_cuda_kernel.cu.o']
    library_dirs = [args.cuda_path, os.path.join(this_file, 'src/')]
    include_dirs = ['/usr/local/cuda-7.5/include/']
else:
    library_dirs = [os.path.join(this_file, 'src/')]
    include_dirs = []

extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

print(torch.cuda.is_available() and args.cuda)

ffi = create_extension(
    '_ext.functions',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=(torch.cuda.is_available() and args.cuda),
    library_dirs=library_dirs,
    include_dirs=include_dirs,
    libraries=libraries,
    extra_compile_args=["-std=gnu11"],
    extra_objects=extra_objects
)

if __name__ == '__main__':
    ffi.build()
