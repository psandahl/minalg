#pragma once

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace minalg {

/**
* @brief Simple and straightforward matrix class. Doubles only.
*/
class Matrix {
public:
    /**
     * @brief Create a zero initialized matrix with given dimensions.
     * @param rows the number of rows.
     * @param columns the number of columns.
    */
    Matrix(std::size_t rows, std::size_t columns);

    /**
     * @brief Create a row matrix from a vector.
     * @param vec the data vector.
    */
    Matrix(const std::vector<double>& vec);

    /**
     * @brief Copy constructor.
     * @param other the Matrix to copy.     
    */
    Matrix(const Matrix& other);

    /**
     * @brief Copy assignment operator.
     * @param other the Matrix to copy.
     * @return this matrix.
    */
    Matrix& operator = (const Matrix& other);

    /**
     * @brief Move constructor.
     * @param other the Matrix to move.   
    */
    Matrix(Matrix&& other);

    /**
     * @brief Move assignment operator.
     * @param other the Matrix to move.
     * @return this matrix.
    */
    Matrix& operator = (Matrix&& other);

    Matrix() = delete;

    /**
     * @brief Destruct the matrix.
    */
    virtual ~Matrix();

    /**
     * @brief Get the number of rows.
     * @return the number of rows.
    */
    std::size_t rows() const { return _rows; }

    /**
     * @brief Get the number of columns.
     * @return the number of columns.
    */    
    std::size_t columns() const { return _columns; }

    /**
     * @brief Get the shape of the matrix.
     * @return tuple<rows, columns>.
    */
    std::tuple<std::size_t, std::size_t> shape() const {
        return std::make_tuple(rows(), columns());
    }

    /**
     * @brief Get the total number of elements.
     * @return the size.
    */
    std::size_t size() const { return rows() * columns(); }

    /**
     * @brief Range checked access to matrix element.
     * @param row the row.
     * @param column the column.
     * @return reference to the requested element.
    */
    double& at(std::size_t row, std::size_t column) {
        if (row < rows() && column < columns()) {
            return _data[linear(row, column)];
        } else {
            throw std::out_of_range("Indexing outside of the matrix");
        }
    }
    const double& at(std::size_t row, std::size_t column) const { 
        if (row < rows() && column < columns()) {
            return _data[linear(row, column)];
        } else {
            throw std::out_of_range("Indexing outside of the matrix");
        }
    }
    
    /**
     * @brief Non range checked access to matrix element.
     * @param row the row.
     * @param column the column.
     * @return reference to the requested element.
    */
    double& get(std::size_t row, std::size_t column) { 
        return _data[linear(row, column)];
    }
    const double& get(std::size_t row, std::size_t column) const { 
        return _data[linear(row, column)];
    }

private:
    std::size_t linear(std::size_t row, std::size_t column) const {
        return row * columns() + column;
    }

    std::size_t _rows;
    std::size_t _columns;
    double *_data;
};

}