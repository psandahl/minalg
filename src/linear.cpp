#include <minalg/linear.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <iostream>

namespace minalg {

static std::size_t find_pivot_index(std::size_t diag, 
                                   const Matrix& m, 
                                   const std::vector<std::size_t>& rows)
{
    std::size_t max_row = diag;
    double max_val = 0.0;

    for (std::size_t row = diag; row < rows.size(); ++row) {
        const double val = std::fabs(m.at(rows[row], diag));
        if (val > max_val) {
            max_row = row;
            max_val = val;
        }
    }

    return max_row;
}

static void multiply_row(Matrix& m, std::size_t row, double value)
{
    for (std::size_t col = 0; col < m.columns(); ++col) {
        m.at(row, col) *= value;
    }
}

static void linearly_combine_rows(Matrix& m, std::size_t dst_row, std::size_t src_row, double factor)
{
    for (std::size_t col = 0; col < m.columns(); ++col) {
        m.at(dst_row, col) += m.at(src_row, col) * factor;
    }
}

Matrix solve(const Matrix& A, const Matrix& b)
{
    if (!A.is_square()) {
        throw std::invalid_argument("Matrix A must be square");
    }
    if (b.columns() != 1 || b.rows() != A.rows()) {
        throw std::invalid_argument("Matrix b must be column matrix with same number of rows as A");
    }

    // Solve the linear system through Gaussian elimination.
    //
    // Augmented matrix.
    minalg::Matrix m(minalg::Matrix::hconcat(A, b));

    // Vector with row indices.
    std::vector<std::size_t> rows(index_vector(A.rows()));

    std::cout << "augmented matrix:\n" << m.info() << std::endl;

    // Iterate through the augmented matrix to get an upper diagonal matrix.
    for (std::size_t diag = 0; diag < rows.size(); ++diag) {
        // Find the pivot index - and swap the indices. The row with the
        // greatest absolute value will now be available in rows[diag].
        const std::size_t pivot_index = find_pivot_index(diag, m, rows);
        std::swap(rows[pivot_index], rows[diag]);

        // Read pivot value.
        const double pivot_value = m.at(rows[diag], diag);
        if (near_zero(pivot_value)) {
            throw std::out_of_range("Singular matrix");
        }

        // Normalize pivot row using pivot value to simplify further
        // calculations.
        multiply_row(m, rows[diag], 1.0 / pivot_value);

        // Eliminate rows below.
        for (std::size_t elim = diag + 1; elim < rows.size(); ++elim) {
            // Read value for the cell to eliminate ...
            const double elim_value = m.at(rows[elim], diag);

            // ... and use the value to make a linear combination that
            // eliminates the cell.
            linearly_combine_rows(m, rows[elim], rows[diag], -elim_value);
        }

        std::cout << "diag=" << diag << ", pivot_value=" << pivot_value << std::endl;
        std::cout << "m=\n" << m.info() << std::endl;
    }

    // Perform back-substitution to get the solutions.
    Matrix x(b.shape());

    for (std::size_t i = x.rows(); i > 0; --i) {
        const std::size_t row = i - 1;
        // Initialize with the corresponding value from the
        // augmented column.        
        double value = m.at(rows[row], m.columns() - 1);

        // Back-substitute.
        for (std::size_t col = row + 1; col < m.columns() - 1; ++col) {
            const double term = -m.at(rows[row], col) * x.at(col, 0);
            value += term;            
        }

        x.at(row, 0) = value;        
    }

    return x;
}

}