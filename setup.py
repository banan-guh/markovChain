from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ext = Pybind11Extension(
    "markov_lib",
    ["wrapper.cpp", "markov.cpp"],
)

setup(name="markov_lib", ext_modules=[ext])