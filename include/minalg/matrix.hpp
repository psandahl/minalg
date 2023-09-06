#pragma once

#include <cstddef>
#include <tuple>

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
    
private:
    const std::size_t _rows;
    const std::size_t _columns;
    double *_data;
};

}