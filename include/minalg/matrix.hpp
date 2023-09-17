#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
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
     * @brief Create a zero initialized matrix with given shape.
     * @param shape the shape.     
    */
    Matrix(const std::tuple<std::size_t, std::size_t>& shape):
        Matrix(std::get<0>(shape), std::get<1>(shape))
    {}

    /**
     * @brief Create a column matrix from a vector.
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
        return { rows(), columns() };
    }

    /**
     * @brief Get the total number of elements.
     * @return the size.
    */
    std::size_t size() const { return rows() * columns(); }

    /**
     * @brief Check whether the matrix is square.
     * @return boolean value.
    */
    bool is_square() const { return rows() == columns(); }

    /**
     * @brief Check whether the matrix is symmetric.
     * @return boolean value.
    */
    bool is_symmetric() const { 
        return is_square() && equal(*this, transpose());    
    }

    /**
     * @brief Check whether the matrix is orthogonal.
     * @return boolean value.
    */
    bool is_orthogonal() const {
        return is_square() && equal(multiply(*this, transpose()), eye(rows()));
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

    /**
     * @brief Range checked access to matrix element.
     * @param row the row.
     * @param column the column.
     * @return reference to the requested element.
    */
    double& at(std::size_t row, std::size_t column) {
        if (row >= rows() || column >= columns()) {
            throw std::out_of_range("Indexing outside of the matrix");
        }

        return get(row, column);
    }
    const double& at(std::size_t row, std::size_t column) const { 
        if (row >= rows() || column >= columns()) {
            throw std::out_of_range("Indexing outside of the matrix");
        }

        return get(row, column);
    }        

    /**
     * @brief Extract the diagonal values.
     * @return vector with values.
    */
    std::vector<double> diag() const { return diag(*this); }

    /**
     * @brief Reshape this matrix.
     * @param shape new shape.
     * @return this matrix.
    */
    Matrix& reshaped(const std::tuple<std::size_t, std::size_t>& shape) {
        reshape(*this, shape);
        return *this;
    }

    /**
     * @brief Create a reshaped copy of this matrix.
     * @param shape new shape.
     * @return reshaped matrix.
    */
    Matrix reshape(const std::tuple<std::size_t, std::size_t>& shape) const {
        Matrix m(*this);
        reshape(m, shape);

        return m;
    }

    /**
     * @brief Create a transposed copy of this matrix.
     * @return transposed matrix.
    */
    Matrix transpose() const { return transpose(*this); }

    /**
     * @brief Multiply the given row with a scale factor.
     * @param row the row number.
     * @param factor the factor to scale the row.
    */
    void scale_row(std::size_t row, double factor);

    /**
     * @brief Linearly combine two rows by adding the scaled source row
     * to the target row.
     * @param src_row the source row.
     * @param factor the factor to scale the source row.
     * @param tgt_row the target row.
    */
    void linearly_combine(std::size_t src_row, double factor, std::size_t tgt_row);

    /**
     * @brief Render a string representation.
     * @param precision the numeric precision.
     * @return the info string.
    */
    std::string info(std::streamsize precision = 5) const;

    /**
     * @brief Create an identity matrix.
     * @param dim the matrix dimension.
     * @return the identity matrix.
    */
    static Matrix eye(std::size_t dim);

    /**
     * @brief Create a diagonal matrix.
     * @param dim the values for the diagonal.
     * @return the diagonal matrix.
    */
    static Matrix diag(const std::vector<double>& vec);

    /**
     * @brief Extract the diagonal values.
     * @return vector with values.
    */
    static std::vector<double> diag(const Matrix& m);

    /**
     * @brief Reshape the matrix to new shape.
     * @param m matrix to reshape.
     * @param shape new shape.
    */
    static void reshape(Matrix& m, const std::tuple<std::size_t, std::size_t>& shape);

    /**
     * @brief Transpose m0 into m1.
     * @param m0 matrix to transpose.
     * @param m1 transposed matrix.
    */
    static void transpose(const Matrix& m0, Matrix& m1);

    /**
     * @brief Transpose the matrix.
     * @return transposed matrix.
    */
    static Matrix transpose(const Matrix& m);

    /**
     * @brief Multiply two matrices.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @param m2 multiplicated matrix.
    */
    static void multiply(const Matrix& m0, const Matrix& m1, Matrix& m2);

    /**
     * @brief Multiply two matrices.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @return multiplicated matrix.
    */
    static Matrix multiply(const Matrix& m0, const Matrix& m1);

    /**
     * @brief Horizontal concat of two matrices.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @param m2 concatenated matrix.
    */
    static void hconcat(const Matrix& m0, const Matrix& m1, Matrix& m2);

    /**
     * @brief Horizontal concat of two matrices.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @return concatenated matrix.
    */
    static Matrix hconcat(const Matrix& m0, const Matrix& m1);

    /**
     * @brief Check whether two matrices are equal.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @return true if equal, false otherwise.
    */
    static bool equal(const Matrix& m0, const Matrix& m1);

private:
    std::size_t linear(std::size_t row, std::size_t column) const {
        return row * columns() + column;
    }

    std::size_t _rows;
    std::size_t _columns;
    double *_data;
};

inline bool operator == (const Matrix& lhs, const Matrix& rhs)
{
    return Matrix::equal(lhs, rhs);
}

inline Matrix operator * (const Matrix& lhs, const Matrix& rhs)
{
    return Matrix::multiply(lhs, rhs);
}

}