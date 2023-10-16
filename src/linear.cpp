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

matrix solve(const matrix& A, const matrix& b)
{
    if (A.is_square()) {
        return lu_solve(lu_decomp(A), b);
    } else if (A.rows() > A.columns() && b.rows() == A.rows() && b.columns() == 1) {
        const matrix At(A.transpose());
        return lu_solve(lu_decomp(At * A), At * b);
    } else {
        throw std::invalid_argument("matrix A must be square or tall");
    }
}

matrix invert(const matrix& A)
{
    const std::tuple<matrix, matrix, matrix> PLU(lu_decomp(A));

    matrix A_inv(A.shape());    
    matrix b(A.rows(), 1);    

    for (std::size_t diag = 0; diag < A_inv.rows(); ++diag) {
        if (diag > 0) {
            b.get(diag - 1, 0) = 0.0;
        }
        b.get(diag, 0) = 1.0;

        const matrix x(lu_solve(PLU, b));
        for (std::size_t row = 0; row < A_inv.rows(); ++row) {
            A_inv.get(row, diag) = x.get(row, 0);
        }
    }

    return A_inv;    
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
    matrix Lp(A.shape());
    matrix Up(A);

    // Vector with row indices.
    std::vector<std::size_t> rows(index_vector(Up.rows()));

    // Iterate the diagonal and find pivot elements in U.
    for (std::size_t diag = 0; diag < rows.size(); ++diag) {
        const std::size_t pivot_row_index = find_pivot_row_index(diag, Up, rows);
        std::swap(rows[pivot_row_index], rows[diag]);        

        // Read the pivot value. Throw if close to zero.
        const double pivot_value = Up.get(rows[diag], diag);
        if (near_zero(pivot_value)) {
            throw singular_matrix("lu_decomp()");
        }

        // Eliminate rows in U, and set cells in L.
        for (std::size_t elim = diag + 1; elim < rows.size(); ++elim) {
            const double elim_value = Up.get(rows[elim], diag) / pivot_value;            

            Lp.get(rows[elim], diag) = elim_value;
            Up.linearly_combine(rows[diag], -elim_value, rows[elim]);
        }
    }

    // Create the permutation matrix P. P is usable for reconstructing
    // A = P * L * U.
    matrix P(A.shape());
    for (std::size_t i = 0; i < rows.size(); ++i) {
        P.get(rows[i], i) = 1.0;
    }

    // But to rectify L and U, P's transpose must be used.
    const matrix Pt(P.transpose());
    matrix L(Pt * Lp);
    const matrix U(Pt * Up);

    // Fill in the diagonal for L.
    for (std::size_t diag = 0; diag < L.rows(); ++diag) {
        L.get(diag, diag) = 1.0;
    }
    
    // Return the matrices.
    return { std::move(P), std::move(L), std::move(U) };
}

matrix lu_solve(const std::tuple<matrix, matrix, matrix>& PLU, const matrix& b)
{
    const auto [P, L, U] = PLU;

    if (!P.is_square() || !L.is_square() || !U.is_square()) {
        throw std::invalid_argument("P, L and U must be square");
    }

    if (P.shape() != L.shape() || P.shape() != U.shape()) {
        throw std::invalid_argument("P, L and U must have same shape");
    }

    if (b.rows() != L.rows() || b.columns() != 1) {
        throw std::invalid_argument("b must be column vector same height as L");
    }

    // Step 1. Forward substitution using L and b: Ly = b.
    // Assume L has ones on the diagonal during the forward substitution.
    matrix Pb(P * b); // Apply permutation on b.
    matrix y(Pb.shape());    

    for (std::size_t diag = 0; diag < L.rows(); ++diag) {
        double value = Pb.get(diag, 0);
        for (std::size_t i = 0; i < diag; ++i) {
            value -= L.get(diag, i) * y.get(i, 0);
        }
        y.get(diag, 0) = value;
    }    

    // Step 2. Backward substitution using U and y: Ux = y.
    matrix x(y.shape());

    for (int diag = U.rows() - 1; diag >= 0; --diag) {        
        double value = y.get(diag, 0);        
        for (int i = U.columns() - 1; i > diag; --i) {            
            value -= U.get(diag, i) * x.get(i, 0);
        }        
        x.get(diag, 0) = value / U.get(diag, diag);
    }    

    return x;
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
    } catch (const singular_matrix&) {
        // The matrix A is singular - the determinant is zero.
        return 0.0;
    }
}

std::tuple<matrix, matrix> qr_decomp(const matrix& A)
{    
    const auto [rows, columns] = A.shape();
    if (rows < columns) {
        throw std::invalid_argument("matrix must be tall or square");
    }

    // Create matrices, where R is a copy of A.
    matrix Q(matrix::eye(rows));
    matrix R(A);

    // Traverse the diagonal of R and calculate the Householder reflector.
    const std::size_t max_diag = A.is_square() ? columns - 1 : columns;
    for (std::size_t diag = 0; diag < max_diag; ++diag) {
        const matrix z(R.slice({diag, diag}, {rows - 1, diag}));

        matrix e(z.shape());
        e.get(0, 0) = 1.0;

        // Normal ...
        matrix n(e * z.norm() * -sign(z.get(0, 0)) - z);
        n *= 1.0 / n.norm();

        // ... and normal transpose.
        const matrix nT(n.transpose());        

        // Apply the Householder reflector to the columns of Q and R.
        for (std::size_t j = 0; j < rows; ++j) {
            const matrix Qval(Q.slice({diag, j}, {rows - 1, j}));
            const matrix nn2(n * (nT * Qval) * 2.0);

            for (std::size_t i = diag, r = 0; i < rows; ++i, ++r) {                
                Q.get(i, j) -= nn2.get(r, 0);
            }
        }

        for (std::size_t j = 0; j < columns; ++j) {
            const matrix Rval(R.slice({diag, j}, {rows - 1, j}));
            const matrix nn2(n * (nT * Rval) * 2.0);
            
            for (std::size_t i = diag, r = 0; i < rows; ++i, ++r) {            
                R.get(i, j) -= nn2.get(r, 0);
            }
        }

        // Zero the cells under the diagonal of R.
        for (std::size_t i = diag + 1; i < rows; ++i) {
            R.get(i, diag) = 0.0;
        }
    }

    // Return the matrices.
    return { std::move(Q.transpose()), std::move(R) };
}

std::vector<double> eigvals(const matrix& A)
{
    if (!A.is_square()) {
        throw std::invalid_argument("matrix must be square");
    }

    matrix m(A);
    static constexpr std::size_t max_iter = 100;
    for (std::size_t i = 0; i < max_iter; ++i) {
        const auto [Q, R] = qr_decomp(m);
        matrix::multiply(R, Q, m); // Save a little copying, at least :-)

        // If the matrix is upper triangular, then the solution has
        // converged.
        if (m.is_upper_triangular()) {            
            break;
        }        
    }

    return m.diag();
}

std::tuple<std::vector<double>, matrix> eig(const matrix& A)
{
    if (!A.is_symmetric()) {
        throw std::invalid_argument("matrix must be symmetric");
    }

    // Calculate the eigenvalues.
    const std::vector<double> evals(eigvals(A));
    
    // Pre-compute initial vector values.
    const std::vector<double> initial(evals.size(), 1.0);

    // Matrix with eigenvector results.
    matrix evecs(A.shape());

    static constexpr std::size_t max_iter = 100;
    static constexpr double threshold = 1e-07;

    // Calculate the eigenvectors using shifted power iterations.
    for (std::size_t i = 0; i < evals.size(); ++i) {
        const double eval = evals[i];

        // Calculate m = A - lambda * eye. Add offset to prevent
        // singular matrix when eigenvalues are subtracted.
        static constexpr double offset = 0.1;
        matrix m(A);
        for (std::size_t diag = 0; diag < m.rows(); ++diag) {
            m.get(diag, diag) -= eval + offset;
        }

        // LU decomposition of m.
        const auto PLU(lu_decomp(m));

        // Set initial vector.
        matrix v(initial);
        v.reshaped({ m.rows(), 1 });
        
        // Refine the initial vector.
        for (std::size_t j = 0; j < max_iter; ++j) {
            // A lot of temporary matrices there is :-(. Perhaps an alternative version
            // of lu_solve, with explicit target matrix?
            matrix tmp(lu_solve(PLU, v));
            tmp *= (1.0 / tmp.norm());

            const matrix diff(abs(tmp) - abs(v));
            v = std::move(tmp);

            if (diff.norm() < threshold) {                
                break;
            }
        }

        // Set result in eigenvector matrix.
        for (std::size_t j = 0; j < v.rows(); ++j) {
            evecs.get(j, i) = v.get(j, 0);
        }
    }

    return { std::move(evals), std::move(evecs) };
}

}}