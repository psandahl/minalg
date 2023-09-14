#pragma once

#include <minalg/matrix.hpp>

namespace minalg {

/**
 * @brief Solve the linear system Ax=b.
 * @param A matrix A.
 * @param b matrix b.
 * @return matrix x.
*/
Matrix solve(const Matrix& A, const Matrix& b);

}