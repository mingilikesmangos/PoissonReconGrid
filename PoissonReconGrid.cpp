// PoissonReconGrid.cpp

#include "Reconstructors.h"
#include "MyMiscellany.h"
#include "PPolynomial.h"
#include "FEMTree.h"
#include "DataStream.imp.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>

#include <cstring>  // For std::memcpy

namespace nb = nanobind;
using namespace nb::literals;

// Define dimension and types
constexpr unsigned int Dim = 3;
using Real = float;
constexpr unsigned int Degree = 2;
constexpr PoissonRecon::BoundaryType BType = PoissonRecon::BOUNDARY_NEUMANN;
constexpr unsigned int FEMSig = PoissonRecon::FEMDegreeAndBType<Degree, BType>::Signature;

// Implement ArrayInputSampleStream class
class ArrayInputSampleStream : public PoissonRecon::Reconstructor::InputSampleStream<Real, Dim> {
public:
    ArrayInputSampleStream(const Real* points, const Real* normals, size_t nPoints)
        : _points(points), _normals(normals), _nPoints(nPoints), _current(0) {}

    void reset(void) override { _current = 0; }

    bool base_read(PoissonRecon::Reconstructor::Position<Real, Dim>& p,
                   PoissonRecon::Reconstructor::Normal<Real, Dim>& n) override {
        if (_current >= _nPoints) return false;
        for (unsigned int d = 0; d < Dim; d++) {
            p[d] = _points[_current * Dim + d];
            n[d] = _normals[_current * Dim + d];
        }
        _current++;
        return true;
    }

    bool base_read(unsigned int thread,
                   PoissonRecon::Reconstructor::Position<Real, Dim>& p,
                   PoissonRecon::Reconstructor::Normal<Real, Dim>& n) override {
        return base_read(p, n);
    }

private:
    const Real* _points;
    const Real* _normals;
    size_t _nPoints;
    size_t _current;
};

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

    // Create the sample stream
    ArrayInputSampleStream sampleStream(points_data, normals_data, nPoints);

    // Set up parameters
    PoissonRecon::Reconstructor::Poisson::SolutionParameters<Real> sParams;
    sParams.depth = depth;
    sParams.pointWeight = 4.0f;
    sParams.verbose = false;

    // Create the implicit function
    PoissonRecon::Reconstructor::Poisson::Implicit<Real, Dim, FEMSig> implicit(sampleStream, sParams);

    // Evaluate the regular grid
    int res = 0;
    bool primalGrid = false;
    Real* values = implicit.tree.template regularGridEvaluate<true>(implicit.solution, res, -1, primalGrid);

    // Create an ndarray directly from the 'values' array
    size_t res_size = static_cast<size_t>(res);
    
    // NB: Mingi, don't forget noexcept here
    nb::capsule owner(values, [](void *data) noexcept {
        delete[] static_cast<Real*>(data);
    });

    auto grid = nb::ndarray<nb::numpy, Real, nb::ndim<3>>(
        /* data = */ values,
        /* shape = */ { res_size, res_size, res_size },
        /* owner = */ owner
        /* NB: If you want to change the stride, do it here! I don't know how what the layout of your data is above ;)  */
    );

    // Return the grid
    return grid;
}

// Expose the function using nanobind
NB_MODULE(poissonrecongrid, m) {
    m.def("poisson_reconstruction_grid", &poisson_reconstruction_grid, "points"_a, "normals"_a, "depth"_a = 8);
}
