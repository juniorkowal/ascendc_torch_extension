import os
import platform
import torch_npu
from setuptools import Extension, setup, find_packages
from torch_npu.utils.cpp_extension import TorchExtension
from torch.utils.cpp_extension import BuildExtension
from pathlib import Path


PACKAGE_NAME = "gather_custom"
VERSION = "1.0.0"
PYTORCH_NPU_INSTALL_PATH = os.path.dirname(os.path.abspath(torch_npu.__file__))
PLATFORM_ARCH = platform.machine() + "-linux"
BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def AscendCExtension(name, sources, extra_library_dirs, extra_libraries, extra_link_args):
    kwargs = {}
    cann_home = os.environ['ASCEND_TOOLKIT_HOME']
    include_dirs = [
            os.path.join(cann_home, PLATFORM_ARCH, 'include'),
            os.path.join(PYTORCH_NPU_INSTALL_PATH, 'include')
            ]
    include_dirs.extend(TorchExtension.include_paths())
    kwargs['include_dirs'] = include_dirs

    library_dirs = [
            os.path.join(cann_home, PLATFORM_ARCH, 'lib64'),
            os.path.join(PYTORCH_NPU_INSTALL_PATH, 'lib'),
            ]
    library_dirs.extend(extra_library_dirs)
    library_dirs.extend(TorchExtension.library_paths())
    kwargs['library_dirs'] = library_dirs

    libraries = [
            'c10', 'torch', 'torch_cpu', 'torch_npu', 'torch_python', 'ascendcl',
            ]
    libraries.extend(extra_libraries)
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    kwargs['extra_link_args'] = extra_link_args
    return Extension(name, sources, **kwargs)

# def get_csrc():
#     csrc_dir = os.path.join(BASE_DIR, "src", PACKAGE_NAME)
#     return sorted([str(p) for p in Path(csrc_dir).rglob("*.cpp")])



if __name__ == "__main__":
    package_data = {
        PACKAGE_NAME: [
            'lib/*.so',
        ]
    }
    setup(
            name=PACKAGE_NAME,
            version=VERSION,
            packages=find_packages(where="src"),
            package_dir={"": "src"},
            package_data=package_data,
            ext_modules=[
                AscendCExtension(
                    name=f"{PACKAGE_NAME}._C",
                    sources=[f"{BASE_DIR}/src/pybind11.cpp"],#get_csrc(),
                    extra_library_dirs=[
                        os.path.join(f"{BASE_DIR}/src/{PACKAGE_NAME}/lib")
                        ],  # location of custom lib{name}.so file
                    extra_libraries=[
                        'gather_custom_ascendc'
                        ],  # name of custom lib{name}.so file
                    extra_link_args=[
                        '-Wl,-rpath,$ORIGIN/lib'
                    ],
                    ),
                ],
            cmdclass={
                'build_ext': BuildExtension
                }
            )

