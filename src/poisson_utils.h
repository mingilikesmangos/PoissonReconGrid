#include <cstring>  // For std::memcpy
#include <vector>

using Real = float;

constexpr unsigned int Degree = 2;
constexpr unsigned int Dim = 3;

int poisson_reconstruction(const Real* points_data, const Real* normals_data, size_t nPoints, std::vector<Real>& _values, int depth = 8);