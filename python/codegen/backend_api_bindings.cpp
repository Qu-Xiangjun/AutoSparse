#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 

#include "backend_api.hpp"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(auto_sparse_backend, m) { // 这里的模块名需要和文件名对应
    py::class_<BackEndAPI>(m, "BackEndAPI")
        .def(py::init<const std::string&, const std::vector<std::string>&>(),
             py::arg("init_info"), py::arg("filepaths"))
        .def("Compute", &BackEndAPI::Compute, 
             py::arg("schedule"), py::arg("warm") = 10, py::arg("round") = 50, py::arg("time_policy") = "avg",
             "Compute function with optional parameters warm, round, and avg_time.")
        .def_readonly("origin_time", &BackEndAPI::origin_time, byref);
}
