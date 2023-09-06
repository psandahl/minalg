#include "minalg/matrix.hpp"

#include <cstring>

namespace minalg {

Matrix::Matrix(std::size_t rows, std::size_t columns):
    _rows(rows),
    _columns(columns),
    _data(nullptr)
{
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("Arguments for both rows and columns must be > 0");
    }

    const std::size_t size = rows * columns;
    _data = new double[size];
    std::memset(_data, 0, sizeof(double) * size);
}

Matrix::Matrix(const Matrix& other):
    _rows(other.rows()),
    _columns(other.columns()),
    _data(new double[other.size()])
{
    std::memcpy(_data, other._data, sizeof(double) * other.size());
}

Matrix& Matrix::operator = (const Matrix& other)
{
    if (this != &other) {
        delete[] _data;

        _rows = other.rows();
        _columns = other.columns();
        _data = new double[other.size()];
        std::memcpy(_data, other._data, sizeof(double) * other.size());
    }    

    return *this;
}

Matrix::Matrix(Matrix&& other):
    _rows(other.rows()),
    _columns(other.columns()),
    _data(new double[other.size()])
{
    std::memcpy(_data, other._data, sizeof(double) * other.size());

    other._rows = 0;
    other._columns = 0;
    delete[] other._data;
    other._data = nullptr;
}

Matrix& Matrix::operator = (Matrix&& other)
{
    if (this != &other) {
        delete[] _data;

        _rows = other.rows();
        _columns = other.columns();
        _data = new double[other.size()];
        std::memcpy(_data, other._data, sizeof(double) * other.size());

        other._rows = 0;
        other._columns = 0;
        delete[] other._data;
        other._data = nullptr;
    }

    return *this;
}

Matrix::~Matrix()
{
    _rows = 0;
    _columns = 0;
    delete[] _data;
    _data = nullptr;
}

}