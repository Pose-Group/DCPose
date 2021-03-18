import os
import platform
import subprocess
import time
from setuptools import Extension, find_packages, setup

import numpy as np
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, sources):
    return CUDAExtension(
        name=name,
        sources=[p for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


if __name__ == '__main__':
    setup(
        name='deform_conv',
        ext_modules=[
            make_cuda_ext(
                name='deform_conv_cuda',
                sources=[
                    'src/deform_conv_cuda.cpp',
                    'src/deform_conv_cuda_kernel.cu'
                ]),
            make_cuda_ext(
                name='deform_pool_cuda',
                sources=[
                    'src/deform_pool_cuda.cpp',
                    'src/deform_pool_cuda_kernel.cu'
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)

