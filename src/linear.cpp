#include <minalg/linear.hpp>
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <cstring>
#include <vector>

namespace minalg {
namespace linear {

static matrix gaussian_elimination(const matrix& A, const matrix& b);
static matrix gauss_jordan_elimination(const matrix& A);

matrix solve(const matrix& A, const matrix& b)
{
    if (!A.is_square()) {
        throw std::invalid_argument("matrix A must be square");
    }
    if (b.columns() != 1 || b.rows() != A.rows()) {
        throw std::invalid_argument("matrix b must be column matrix with same number of rows as A");
    }

    return gaussian_elimination(A, b);
}

matrix invert(const matrix& A)
{
    if (!A.is_square()) {
        throw std::invalid_argument("matrix must be square");
    }

    return gauss_jordan_elimination(A);
}

std::tuple<matrix, matrix, matrix> lu_decomp(const matrix& A)
{
    // This implementation requires square matrices, even though
    // the decomposition of non-square matrices are possible.
    if (!A.is_square()) {
        throw std::invalid_argument("matrix A must be square");
    }

    // Permuted L and U matrices. As the last step the matrices
    // will be rectified to their normal form.
    minalg::matrix Lp(A.shape());
    minalg::matrix Up(A);

    // Vector with row indices.
    std::vector<std::size_t> rows(index_vector(Up.rows()));

    // Iterate the diagonal and find pivot elements in U.
    for (std::size_t diag = 0; diag < rows.size() - 1; ++diag) {
        const std::size_t pivot_row_index = find_pivot_row_index(diag, Up, rows);
        std::swap(rows[pivot_row_index], rows[diag]);        

        // Read the pivot value. Throw if close to zero.
        const double pivot_value = Up.at(rows[diag], diag);
        if (near_zero(pivot_value)) {
            throw std::invalid_argument("Singular matrix");
        }

        // Eliminate rows in U, and set cells in L.
        for (std::size_t elim = diag + 1; elim < rows.size(); ++elim) {
            const double elim_value = Up.at(rows[elim], diag) / pivot_value;            

            Lp.get(rows[elim], diag) = elim_value;
            Up.linearly_combine(rows[diag], -elim_value, rows[elim]);
        }
    }

    // Create the permutation matrix P. P is usable for reconstructing
    // A = P * L * U.
    minalg::matrix P(A.shape());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        P.get(rows[i], i) = 1.0;
    }

    // But to rectify L and U, P's transpose must be used.
    const minalg::matrix Pt(P.transpose());
    minalg::matrix L(Pt * Lp);
    const minalg::matrix U(Pt * Up);

    // Fill in the diagonal for L.
    for (std::size_t diag = 0; diag < L.rows(); ++diag) {
        L.get(diag, diag) = 1.0;
    }
    
    // Return the matrices.
    return { std::move(P), std::move(L), std::move(U) };
}

double det(const matrix& A)
{
    if (!A.is_square()) {
        throw std::invalid_argument("matrix A must be square");
    }

    try {
        const auto [P, L, U] = lu_decomp(A);

        double sum = 0.0;
        double det = 1.0;
        for (std::size_t diag = 0; diag < P.rows(); ++diag) {
            sum += P.get(diag, diag);
            det *= U.get(diag, diag);
        }

        // The sign of the determinant changes with permutation.
        const long diff = P.rows() - static_cast<long>(sum);
        const long changed_rows = diff - 1;
        const double sign = diff == 0 || changed_rows % 2 == 0 ? 1.0 : -1.0;

        return sign * det;
    } catch (const std::invalid_argument&) {
        // The matrix A is singular - the determinant is zero.
        return 0.0;
    }
}

matrix gaussian_elimination(const matrix& A, const matrix& b)
{
    // Solve the linear system through Gaussian elimination.
    //
    // Augmented matrix.
    minalg::matrix m(minalg::matrix::hconcat(A, b));

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
            throw std::invalid_argument("Singular matrix");
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
    matrix x(b.shape());

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

matrix gauss_jordan_elimination(const matrix& A)
{
    // Invert the matrix A through Gauss Jordan elimination.
    //
    // Augmented matrix.
    minalg::matrix m(minalg::matrix::hconcat(A, minalg::matrix::eye(A.rows())));

    // Vector with row indices.
    std::vector<std::size_t> rows(index_vector(A.rows()));

    // Iterate through the diagonal of the A part of the augmented matrix.
    for (std::size_t diag = 0; diag < rows.size(); ++diag) {
        // Find the pivot index - and swap the indices. The row with the
        // greatest absolute value will now be available in rows[diag].
        const std::size_t pivot_row_index = find_pivot_row_index(diag, m, rows);
        std::swap(rows[pivot_row_index], rows[diag]);

        // Read pivot value.
        const double pivot_value = m.at(rows[diag], diag);
        if (near_zero(pivot_value)) {
            throw std::invalid_argument("Singular matrix");
        }

        // Normalize pivot row using pivot value to simplify further
        // calculations.        
        m.scale_row(rows[diag], 1. / pivot_value);

        // Eliminate rows above and below.
        for (std::size_t elim = 0; elim < rows.size(); ++elim) {
            if (rows[elim] != rows[diag]) {                
                // Read value for the cell to eliminate ...
                const double elim_value = m.get(rows[elim], diag);                

                // ... and use the value to make a linear combination that
                // eliminates the cell.
                m.linearly_combine(rows[diag], -elim_value, rows[elim]);
            }
        }        
    }

    // Fill in the matrix Ainv.
    minalg::matrix Ainv(A.shape());
    for (std::size_t row = 0; row < A.rows(); ++row) {
        std::memcpy(&Ainv.at(row, 0), &m.at(rows[row], A.columns()), 
                    A.columns() * sizeof(double));
    }

    return Ainv;
}

}}