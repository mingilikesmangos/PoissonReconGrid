#include <Reconstructors.h>
#include <MyMiscellany.h>
#include <PPolynomial.h>
#include <FEMTree.h>
#include <DataStream.imp.h>

#include <cstring>  // For std::memcpy
#include <vector>

#include "poisson_utils.h"

constexpr BoundaryType BType = BOUNDARY_NEUMANN;
constexpr unsigned int FEMSig = FEMDegreeAndBType<Degree, BType>::Signature;

// Implement ArrayInputSampleStream class
class ArrayInputSampleStream : public Reconstructor::InputSampleStream<Real, Dim> {
public:
    ArrayInputSampleStream(const Real* points, const Real* normals, size_t nPoints)
        : _points(points), _normals(normals), _nPoints(nPoints), _current(0) {}

    void reset(void) override { _current = 0; }

    bool base_read(Reconstructor::Position<Real, Dim>& p,
                   Reconstructor::Normal<Real, Dim>& n) override {
        if (_current >= _nPoints) return false;
        for (unsigned int d = 0; d < Dim; d++) {
            p[d] = _points[_current * Dim + d];
            n[d] = _normals[_current * Dim + d];
        }
        _current++;
        return true;
    }

    // bool base_read(unsigned int thread,
    //                Reconstructor::Position<Real, Dim>& p,
    //                Reconstructor::Normal<Real, Dim>& n) override {
    //     return base_read(p, n);
    // }

private:
    const Real* _points;
    const Real* _normals;
    size_t _nPoints;
    size_t _current;
};

int poisson_reconstruction(const Real* points_data, const Real* normals_data, size_t nPoints, std::vector<Real>& _values, int depth) {
    // Create the sample stream
    ArrayInputSampleStream sampleStream(points_data, normals_data, nPoints);

    // Set up parameters
    Reconstructor::Poisson::SolutionParameters<Real> sParams;
    sParams.depth = depth;
    sParams.pointWeight = 4.0f;
    sParams.verbose = false;

    // Create the implicit function
    Reconstructor::Poisson::Implicit<Real, Dim, FEMSig> implicit(sampleStream, sParams);

    // Evaluate the regular grid
    int res = 0;
    bool primalGrid = false;
    Real* values = implicit.tree.template regularGridEvaluate<true>(implicit.solution, res, -1, primalGrid);

    // Convert to vector and return
    _values = std::vector<Real>(values, values + (res * res * res));
    return res;
}