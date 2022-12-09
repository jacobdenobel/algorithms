#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

py::array_t<double> get_t(const std::vector<double>& omega, const size_t n) {
    const auto N = n * n;
    std::vector<double> T(N, 1);
    size_t o = 0;
    for (auto i = 0; i < n-1; i++){
        for (auto j = i + 1; j < n; j++){
            const double angle = omega[o++];
            const double co = std::cos(angle);
            const double so = std::sin(angle);
            const auto i1 = i + n * i;
            const auto i2 = j + n * j;
            const auto i3 = i + n * j;
            const auto i4 = j + n * i;

            T[i1] = T[i1] * co;
            T[i2] = T[i2] * co;
            T[i3] = T[i3] * -so;
            T[i4] = T[i4] * so;
        }
    }    
    return py::array_t<double>(T.size(), T.data());
}

PYBIND11_MODULE(escpp, m)
{
    m.def("get_t", &get_t,py::arg("omega"), py::arg("n"));
}