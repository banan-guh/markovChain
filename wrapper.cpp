#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "markov.h"

namespace py = pybind11;

// The name 'markov_lib' must match your compiled filename (.pyd or .so)
PYBIND11_MODULE(markov_lib, m) {
  py::class_<Markov>(m, "MarkovBot")
    .def(py::init<>())
    .def("train", &Markov::train)
    .def("train_from_file", &Markov::train_from_file)
    .def("generate", &Markov::generate)
    .def("generate_seeded", &Markov::generate_seeded)
    .def("save", &Markov::save_brain)
    .def("load", &Markov::load_brain)
    .def("sanitize", &Markov::sanitize);
}