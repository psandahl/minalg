#pragma once

#include <minalg/matrix.hpp>

namespace minalg {

/**
 * Compute the inner product between a row in matrix m0 and
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

}