#include <minalg/matrix.hpp>
#include "util.hpp"

#include <cstring>
#include <sstream>

namespace minalg {

matrix::matrix(std::size_t rows, std::size_t columns):
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

matrix::matrix(const std::vector<double>& vec):
    _rows(vec.size()),
    _columns(1),
    _data(new double[vec.size()])
{
    std::memcpy(_data, vec.data(), sizeof(double) * vec.size());
}

matrix::matrix(const matrix& other):
    _rows(other.rows()),
    _columns(other.columns()),
    _data(new double[other.size()])
{
    std::memcpy(_data, other._data, sizeof(double) * other.size());
}

matrix& matrix::operator = (const matrix& other)
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

matrix::matrix(matrix&& other):
    _rows(other.rows()),
    _columns(other.columns()),
    _data(other._data)
{
    other._rows = 0;
    other._columns = 0;
    other._data = nullptr;
}

matrix& matrix::operator = (matrix&& other)
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

matrix::~matrix()
{
    _rows = 0;
    _columns = 0;
    delete[] _data;
    _data = nullptr;
}

std::string matrix::info(std::streamsize precision) const
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

matrix matrix::eye(std::size_t dim)
{
    matrix m(dim, dim);

    for (std::size_t i = 0; i < dim; ++i) {
        m.get(i, i) = 1.0;
    }

    return m;
}

matrix matrix::diag(const std::vector<double>& vec)
{
    const std::size_t dim = vec.size();
    matrix m(dim, dim);

    for (std::size_t i = 0; i < dim; ++i) {
        m.get(i, i) = vec[i];
    }

    return m;
}

std::vector<double> matrix::diag(const matrix& m)
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

void matrix::reshape(matrix& m, const shape_t& shape)
{
    const auto [rows, columns] = shape;
    if (m.size() != rows * columns) {
        throw std::invalid_argument("Can only reshape to equal and valid size");
    }

    m._rows = rows;
    m._columns = columns;    
}

void matrix::transpose(const matrix& m0, matrix& m1)
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

matrix matrix::transpose(const matrix& m)
{
    matrix m1(m.columns(), m.rows());
    transpose(m, m1);

    return m1;
}

void matrix::scale_row(std::size_t row, double factor)
{
    double *ptr = &at(row, 0);
    for (std::size_t col = 0; col < columns(); ++col) {
        *ptr++ *= factor;
    }
}

void matrix::linearly_combine(std::size_t src_row, double factor, std::size_t tgt_row)
{
    double *src = &at(src_row, 0);
    double *tgt = &at(tgt_row, 0);

    for (std::size_t col = 0; col < columns(); ++col) {
        *tgt++ += *src++ * factor;
    }
}

void matrix::multiply(const matrix& m0, const matrix& m1, matrix& m2)
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
    
matrix matrix::multiply(const matrix& m0, const matrix& m1)
{
    matrix m2(m0.rows(), m1.columns());
    multiply(m0, m1, m2);

    return m2;
}

void matrix::hconcat(const matrix& m0, const matrix& m1, matrix& m2)
{
    const auto [rows0, columns0] = m0.shape();
    const auto [rows1, columns1] = m1.shape();
    const auto [rows2, columns2] = m2.shape();
    if (rows0 != rows1 || rows1 != rows2) {
        throw std::invalid_argument("m0, m1 and m2 does not match for horizontal concat");
    }
    if (columns2 != columns0 + columns1) {
        throw std::invalid_argument("m0, m1 and m2 does not match for horizontal concat");
    }

    for (std::size_t row = 0; row < rows0; ++row) {
        double *dst = &m2.at(row, 0);
        const double *src = &m0.get(row, 0);
        std::memcpy(dst, src, columns0 * sizeof(double));

        dst += columns0;
        src = &m1.get(row, 0);
        std::memcpy(dst, src, columns1 * sizeof(double));
    }
}

matrix matrix::hconcat(const matrix& m0, const matrix& m1)
{
    matrix m2(m0.rows(), m0.columns() + m1.columns());
    hconcat(m0, m1, m2);

    return m2;
}

bool matrix::equal(const matrix& m0, const matrix& m1)
{
    if (&m0 == &m1) {
        return true;
    }

    if (m0.shape() == m1.shape()) {
        const double* m0_ptr = &m0.get(0, 0);
        const double* m1_ptr = &m1.get(0, 0);
        for (std::size_t i = 0; i < m0.size(); ++i) {
            const double diff = *m0_ptr++ - *m1_ptr++;
            if (!near_zero(diff)) {
                return false;
            }
        }        

        return true;
    } else {
        return false;
    }
}

}