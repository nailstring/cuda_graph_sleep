from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sleep_ext = CUDAExtension(
    'cuda_graph_utils.sleep_ops', 
    ['sleep.cpp', 'sleep_kernel.cu']
)

setup(
    name='cuda_graph_utils',
    ext_modules=[
        sleep_ext,
    ],
    cmdclass={
        'build_ext': BuildExtension,
    }
)
