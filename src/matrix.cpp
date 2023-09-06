#include "minalg/matrix.hpp"

#include <cstring>
#include <stdexcept>

namespace minalg {

Matrix::Matrix(std::size_t rows, std::size_t columns):
    _rows(rows),
    _columns(columns),
    _data(nullptr)
{
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("Arguments for both rows and columns must be > 0");
    }

    const std::size_t amount = rows * columns;
    _data = new double[amount];
    std::memset(_data, 0, sizeof(double) * amount);
}

Matrix::~Matrix()
{
    delete[] _data;
    _data = nullptr;
}

}