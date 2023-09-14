#include <minalg/matrix.hpp>
#include "util.hpp"

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
    _rows(vec.size()),
    _columns(1),
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

    for (std::size_t i = 0; i < dim; ++i) {
        m.get(i, i) = 1.0;
    }

    return m;
}

Matrix Matrix::diag(const std::vector<double>& vec)
{
    const std::size_t dim = vec.size();
    Matrix m(dim, dim);

    for (std::size_t i = 0; i < dim; ++i) {
        m.get(i, i) = vec[i];
    }

    return m;
}

std::vector<double> Matrix::diag(const Matrix& m)
{
    if (!m.is_square()) {
        throw std::invalid_argument("Attempting to extract diagonal from non-square matrix");
    }

    std::vector<double> vec(m.rows());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        vec[i] = m.get(i, i);
    }
    
    return vec;
}

void Matrix::reshape(Matrix& m, const std::tuple<std::size_t, std::size_t>& shape)
{
    const auto [rows, columns] = shape;
    if (m.size() != rows * columns) {
        throw std::invalid_argument("Can only reshape to equal and valid size");
    }

    m._rows = rows;
    m._columns = columns;    
}

void Matrix::transpose(const Matrix& m0, Matrix& m1)
{
    const auto [rows0, columns0] = m0.shape();
    const auto [rows1, columns1] = m1.shape();
    if (rows0 != columns1 || rows1 != columns0) {
        throw std::invalid_argument("Non matching dimensions for transpose");
    }

    for (std::size_t i = 0; i < columns0; ++i) {
        for (std::size_t j = 0; j < rows0; ++j) {
            m1.get(i, j) = m0.get(j, i);
        }
    }
}

Matrix Matrix::transpose(const Matrix& m)
{
    Matrix m1(m.columns(), m.rows());
    transpose(m, m1);

    return m1;
}

void Matrix::multiply(const Matrix& m0, const Matrix& m1, Matrix& m2)
{
    const auto [rows0, columns0] = m0.shape();
    const auto [rows1, columns1] = m1.shape();
    const auto [rows2, columns2] = m2.shape();
    if (columns0 != rows1) {
        throw std::invalid_argument("m0 and m1 does not match for multiplication");
    }
    if (rows2 != rows0 || columns2 != columns1) {
        throw std::invalid_argument("m2 does not match m0 and m1 for multiplication");
    }

    const std::size_t len = columns0;
    for (std::size_t row = 0; row < rows2; ++row) {        
        for (std::size_t column = 0; column < columns2; ++column) {
            m2.get(row, column) = inner_product(m0, m1, row, column, len);
        }
    }
}
    
Matrix Matrix::multiply(const Matrix& m0, const Matrix& m1)
{
    Matrix m2(m0.rows(), m1.columns());
    multiply(m0, m1, m2);

    return m2;
}

}