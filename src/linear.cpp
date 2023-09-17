#include <minalg/linear.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace minalg {

static Matrix gaussian_elimination(const Matrix& A, const Matrix& b);

Matrix solve(const Matrix& A, const Matrix& b)
{
    if (!A.is_square()) {
        throw std::invalid_argument("Matrix A must be square");
    }
    if (b.columns() != 1 || b.rows() != A.rows()) {
        throw std::invalid_argument("Matrix b must be column matrix with same number of rows as A");
    }

    return gaussian_elimination(A, b);
}

Matrix gaussian_elimination(const Matrix& A, const Matrix& b)
{
    // Solve the linear system through Gaussian elimination.
    //
    // Augmented matrix.
    minalg::Matrix m(minalg::Matrix::hconcat(A, b));

    // Vector with row indices.
    std::vector<std::size_t> rows(index_vector(A.rows()));

    // Iterate through the augmented matrix to get an upper diagonal matrix.
    for (std::size_t diag = 0; diag < rows.size(); ++diag) {
        // Find the pivot index - and swap the indices. The row with the
        // greatest absolute value will now be available in rows[diag].
        const std::size_t pivot_row_index = find_pivot_row_index(diag, m, rows);
        std::swap(rows[pivot_row_index], rows[diag]);

        // Read pivot value.
        const double pivot_value = m.at(rows[diag], diag);
        if (near_zero(pivot_value)) {
            throw std::out_of_range("Singular matrix");
        }

        // Normalize pivot row using pivot value to simplify further
        // calculations.        
        m.scale_row(rows[diag], 1. / pivot_value);

        // Eliminate rows below.
        for (std::size_t elim = diag + 1; elim < rows.size(); ++elim) {
            // Read value for the cell to eliminate ...
            const double elim_value = m.get(rows[elim], diag);

            // ... and use the value to make a linear combination that
            // eliminates the cell.
            m.linearly_combine(rows[diag], -elim_value, rows[elim]);            
        }        
    }

    // Perform back-substitution to get the solutions.
    Matrix x(b.shape());

    for (std::size_t i = x.rows(); i > 0; --i) {
        const std::size_t row = i - 1;
        // Initialize with the corresponding value from the
        // augmented column.        
        double value = m.get(rows[row], m.columns() - 1);

        // Back-substitute.
        for (std::size_t col = row + 1; col < m.columns() - 1; ++col) {
            const double term = -m.get(rows[row], col) * x.get(col, 0);
            value += term;            
        }

        x.get(row, 0) = value;        
    }

    return x;
}

}