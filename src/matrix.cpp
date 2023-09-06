#include "minalg/matrix.hpp"

#include <cstring>
#include <sstream>

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

Matrix::Matrix(const std::vector<double>& vec):
    _rows(1),
    _columns(vec.size()),
    _data(new double[vec.size()])
{
    std::memcpy(_data, vec.data(), sizeof(double) * vec.size());
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
    _data(other._data)
{
    other._rows = 0;
    other._columns = 0;
    other._data = nullptr;
}

Matrix& Matrix::operator = (Matrix&& other)
{
    if (this != &other) {
        delete[] _data;

        _rows = other.rows();
        _columns = other.columns();
        _data = other._data;        

        other._rows = 0;
        other._columns = 0;        
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

std::string Matrix::info(std::streamsize precision) const
{
    std::stringstream ss;
    ss.precision(precision);

    ss << "[\n";
    for (std::size_t row = 0; row < rows(); ++row) {
        ss << "  ";
        for (std::size_t column = 0; column < columns(); ++column) {
            ss << get(row, column);
            if (column < columns() - 1) {
                ss << ", ";
            } else {
                ss << '\n';
            }
        }
    }
    ss << "]" << std::endl;

    return ss.str();
}

Matrix Matrix::eye(std::size_t dim)
{
    Matrix m(dim, dim);

    for (int i = 0; i < dim; ++i) {
        m.get(i, i) = 1.0;
    }

    return m;
}

}