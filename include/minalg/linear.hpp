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

/**
 * @brief Perform LU decomposition of the (square) matrix A into the matrices
 * P (permutation matrix), L (lower triangular matrix) and 
 * U (upper triangular matrix).
 * @param A the matrix to decompose.
 * @return tuple<P, L, U>.
*/
std::tuple<matrix, matrix, matrix> lu_decomp(const matrix& A);

}}