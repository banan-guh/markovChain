from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension(
    "markov_lib",
    ["wrapper.cpp", "markov.cpp"],
)

setup(
    name="markov_lib",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    py_modules=[],
)