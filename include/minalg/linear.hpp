#pragma once

#include <minalg/matrix.hpp>

#include <tuple>

namespace minalg {
namespace linear {

/**
 * @brief Solve the linear system Ax=b.
 * @param A matrix A.
 * @param b matrix b.
 * @return matrix x.
*/
matrix solve(const matrix& A, const matrix& b);

/**
 * @brief Invert the matrix A.
 * @return inverted matrix.
*/
matrix invert(const matrix& A);

std::tuple<matrix, matrix, matrix> lu_decomp(const matrix& A);

}}