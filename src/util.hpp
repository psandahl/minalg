#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace minalg {

class Matrix;

/**
 * @brief Check if a value is close to zero
 * @param value the value to check.
 * @return boolean value.
*/
inline bool near_zero(double value) { return std::fabs(value) < 1e-07; }

/**
 * @brief Compute the inner product between a row in matrix m0 and
 * a column in matrix m1.
 * @param m0 matrix m0.
 * @param m1 matrix m1.
 * @param row_m0 the row in matrix m0.
 * @param col_m1 the column in matrix m1.
 * @param len length of the data.
 * @return the inner product.
*/
double inner_product(const Matrix& m0, const Matrix& m1,
                     std::size_t row_m0, std::size_t col_m1, 
                     std::size_t len);

/**
 * @brief Generate an index vector.
 * @param indices the number of indices.
 * @return the vector with indices [0, indices - 1].
*/
std::vector<std::size_t> index_vector(std::size_t indices);

/**
 * @brief Find the row index with biggest absolute value
 * in the given column.
 * @param diag the current diagonal position (both row and col).
 * @param m the matrix to search within.
 * @param rows the row indices vector.
 * @return the row index with biggest absolute value.
*/
std::size_t find_pivot_row_index(std::size_t diag, 
                                 const Matrix& m, 
                                 const std::vector<std::size_t>& rows);

}