#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace minalg {

class matrix;

/**
 * @brief Check if a value is close to zero
 * @param value the value to check.
 * @return boolean value.
*/
inline bool near_zero(double value, double eps=1e-07) 
{ 
    return std::fabs(value) < eps; 
}

/**
 * @brief Extract the sign from the value.
 * @return the sign for the value (-1 if negative, 1 if positive. 0 if zero).
*/
inline double sign(double value)
{
    if (near_zero(value, 1e-09)) return 0.0;
    return value < 0.0 ? -1.0 : 1.0;
}

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
double inner_product(const matrix& m0, const matrix& m1,
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
                                 const matrix& m, 
                                 const std::vector<std::size_t>& rows);

}