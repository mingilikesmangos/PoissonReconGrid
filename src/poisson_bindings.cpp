#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <cstring>  // For std::memcpy

#include "poisson_utils.h"

namespace nb = nanobind;
using namespace nb::literals;

// Implement poisson_reconstruction_grid function
nb::ndarray<nb::numpy, Real, nb::ndim<3>>poisson_reconstruction_grid(
    nb::ndarray<nb::numpy, Real> points,
    nb::ndarray<nb::numpy, Real> normals,
    int depth = 8
) {
    // Check input arrays
    if (points.ndim() != 2 || normals.ndim() != 2)
        throw std::runtime_error("points and normals must be 2D arrays");
    if (points.shape(0) != normals.shape(0))
        throw std::runtime_error("points and normals must have the same number of rows");
    if (points.shape(1) != Dim || normals.shape(1) != Dim)
        throw std::runtime_error("points and normals must have shape (N, 3)");

    size_t nPoints = points.shape(0);

    // Get pointers to data
    const Real* points_data = static_cast<const Real*>(points.data());
    const Real* normals_data = static_cast<const Real*>(normals.data());

    std::vector<Real> values;

    int res = poisson_reconstruction(points_data, normals_data, nPoints, values, depth);
    // Create an ndarray directly from the 'values' array
    size_t res_size = static_cast<size_t>(res);
    
    // NB: Mingi, don't forget noexcept here
    nb::capsule owner(values.data(), [](void *data) noexcept {
        delete[] static_cast<float *>(data);
    });

    auto grid = nb::ndarray<nb::numpy, Real, nb::ndim<3>>(
        /* data = */ values.data(),
        /* shape = */ { res_size, res_size, res_size },
        /* owner = */ owner
        /* NB: If you want to change the stride, do it here! I don't know how what the layout of your data is above ;)  */
    );

    // Return the grid
    return grid;
}

// Expose the function using nanobind
NB_MODULE(poissonrecongrid_bindings, m) {
    m.def("poisson_reconstruction_grid", &poisson_reconstruction_grid, "points"_a, "normals"_a, "depth"_a = 8);
}
