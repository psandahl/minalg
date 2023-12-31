#pragma once

#include <minalg/matrix.hpp>

#include <stdexcept>
#include <tuple>
#include <vector>

namespace minalg {
namespace linear {

/**
 * @brief Custom runtime error - singular matrix.
*/
class singular_matrix: public std::runtime_error {
public:
    singular_matrix(const std::string& what_arg):
        std::runtime_error(what_arg) {}
    singular_matrix(const char* what_arg):
        std::runtime_error(what_arg) {}

    virtual const char* what() const noexcept {
        return std::runtime_error::what();
    }
};

/**
 * @brief Solve the system Ax = b or AtAx = Atb (i.e. overdetermined system).
 * @param A square or tall matrix A.
 * @param b single column matrix b.
 * @return single column matrix x with solution.
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

/**
 * @brief Solve the linear system P(LU)x=b. P, L and U can be made from
 * square matrix A through lu_decomp.
 * @param PLU tuple with matrices P, L and U.
 * @param b single column matrix b.
 * @return single column matrix x.
*/
matrix lu_solve(const std::tuple<matrix, matrix, matrix>& PLU, const matrix& b);

/**
 * @brief Calculate the determinant of the square matrix A.
 * @param A matrix A.
 * @return the determinant.
*/
double det(const matrix& A);

/**
 * @brief Perform QR decomposition of the (square or tall) matrix A into the 
 * matrices Q (orthogonal matrix) and R (upper triangular matrix).
 * @param A the matrix to decompose.
 * @return tuple<Q, R>.
*/
std::tuple<matrix, matrix> qr_decomp(const matrix& A);

/**
 * @brief Calculate the eigenvalues of the square matrix A. Most exact
 * estimation is given for a symmetric matrix (only those are guaranteed 
 * real eigen values).
 * @param A matrix A.
 * @return vector with eigenvalues, sorted in descending order.
*/
std::vector<double> eigvals(const matrix& A);

/**
 * @brief Calculate the eigenvalues and eigenvectors of the square and
 * symmetric matrix A.
 * @param A matrix A.
 * @return tuple with eigenvalues, sorted in descending order, and matrix
 * with corresponding eigenvectors.
*/
std::tuple<std::vector<double>, matrix> eig(const matrix& A);

}}