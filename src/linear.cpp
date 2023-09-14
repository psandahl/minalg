#include <minalg/linear.hpp>

#include <stdexcept>

namespace minalg {

Matrix solve(const Matrix& A, const Matrix& b)
{
    if (!A.is_square()) {
        throw std::invalid_argument("Matrix A must be square");
    }
    if (b.columns() != 1 || b.rows() != A.rows()) {
        throw std::invalid_argument("Matrix b must be column matrix with same number of rows as A");
    }

    Matrix x(b.shape());

    return x;
}

}