from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

__version__ = "0.0.1"

setup(
    name="es",
    version=__version__,
    description="es",
    long_description="",
    ext_modules=[
        Pybind11Extension("escpp", ["es.cpp"], include_dirs=["."]),
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)