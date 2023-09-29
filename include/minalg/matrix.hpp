#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace minalg {

/**
 * @brief Shape type (alias for tuple with two size_t - rows and columns).
*/
typedef std::tuple<std::size_t, std::size_t> shape_t;

/**
* @brief Simple and straightforward matrix class. Doubles only.
*/
class matrix {
public:
    /**
     * @brief Create a zero initialized matrix with given dimensions.
     * @param rows the number of rows.
     * @param columns the number of columns.
    */
    matrix(std::size_t rows, std::size_t columns);

    /**
     * @brief Create a zero initialized matrix with given shape.
     * @param shape the shape.     
    */
    matrix(const shape_t& shape):
        matrix(std::get<0>(shape), std::get<1>(shape))
    {}

    /**
     * @brief Create a column matrix from a vector.
     * @param vec the data vector.
    */
    matrix(const std::vector<double>& vec);

    /**
     * @brief Copy constructor.
     * @param other the matrix to copy.     
    */
    matrix(const matrix& other);

    /**
     * @brief Copy assignment operator.
     * @param other the matrix to copy.
     * @return this matrix.
    */
    matrix& operator = (const matrix& other);

    /**
     * @brief Move constructor.
     * @param other the matrix to move.   
    */
    matrix(matrix&& other);

    /**
     * @brief Move assignment operator.
     * @param other the matrix to move.
     * @return this matrix.
    */
    matrix& operator = (matrix&& other);

    matrix() = delete;

    /**
     * @brief Destruct the matrix.
    */
    virtual ~matrix();

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
    shape_t shape() const {
        return { rows(), columns() };
    }

    /**
     * @brief Get the total number of elements.
     * @return the size.
    */
    std::size_t size() const { return rows() * columns(); }

    /**
     * @brief Get the L2 norm of a column or row matrix.
     * @return the L2 norm.
    */
    double norm() const;

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
    matrix& reshaped(const shape_t& shape) {
        reshape(*this, shape);
        return *this;
    }

    /**
     * @brief Create a reshaped copy of this matrix.
     * @param shape new shape.
     * @return reshaped matrix.
    */
    matrix reshape(const shape_t& shape) const {
        matrix m(*this);
        reshape(m, shape);

        return m;
    }

    /**
     * @brief Create a transposed copy of this matrix.
     * @return transposed matrix.
    */
    matrix transpose() const { return transpose(*this); }

    /**
     * @brief Slice this matrix.     
     * @param upper_left shape describing the upper leftmost point for slicing.
     * @param lower_right shape describing the lower rightmost point.
     * @return the sliced copy.
    */
    matrix slice(const shape_t& upper_left, const shape_t& lower_right) const {
        return slice(*this, upper_left, lower_right);
    }

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
    static matrix eye(std::size_t dim);

    /**
     * @brief Create a diagonal matrix.
     * @param dim the values for the diagonal.
     * @return the diagonal matrix.
    */
    static matrix diag(const std::vector<double>& vec);

    /**
     * @brief Extract the diagonal values.
     * @return vector with values.
    */
    static std::vector<double> diag(const matrix& m);

    /**
     * @brief Reshape the matrix to new shape.
     * @param m matrix to reshape.
     * @param shape new shape.
    */
    static void reshape(matrix& m, const shape_t& shape);

    /**
     * @brief Transpose m0 into m1.
     * @param m0 matrix to transpose.
     * @param m1 transposed matrix.
    */
    static void transpose(const matrix& m0, matrix& m1);

    /**
     * @brief Transpose the matrix.
     * @return transposed matrix.
    */
    static matrix transpose(const matrix& m);

    /**
     * @brief Multiply two matrices.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @param m2 multiplicated matrix.
    */
    static void multiply(const matrix& m0, const matrix& m1, matrix& m2);

    /**
     * @brief Multiply two matrices.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @return multiplicated matrix.
    */
    static matrix multiply(const matrix& m0, const matrix& m1);

    /**
     * @brief Horizontal concat of two matrices.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @param m2 concatenated matrix.
    */
    static void hconcat(const matrix& m0, const matrix& m1, matrix& m2);

    /**
     * @brief Horizontal concat of two matrices.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @return concatenated matrix.
    */
    static matrix hconcat(const matrix& m0, const matrix& m1);
    
    /**
     * @brief Slice the given matrix.
     * @param m the matrix to slice.
     * @param upper_left shape describing the upper leftmost point for slicing.
     * @param lower_right shape describing the lower rightmost point.
     * @return the sliced copy.
    */
    static matrix slice(const matrix& m, const shape_t& upper_left, 
                        const shape_t& lower_right);

    /**
     * @brief Check whether two matrices are equal.
     * @param m0 left input matrix.
     * @param m1 right input matrix.
     * @return true if equal, false otherwise.
    */
    static bool equal(const matrix& m0, const matrix& m1);

private:
    std::size_t linear(std::size_t row, std::size_t column) const {
        return row * columns() + column;
    }

    std::size_t _rows;
    std::size_t _columns;
    double *_data;
};

inline bool operator == (const matrix& lhs, const matrix& rhs)
{
    return matrix::equal(lhs, rhs);
}

inline bool operator != (const matrix& lhs, const matrix& rhs)
{
    return !matrix::equal(lhs, rhs);
}

inline matrix operator * (const matrix& lhs, const matrix& rhs)
{
    return matrix::multiply(lhs, rhs);
}

inline matrix operator * (const matrix& lhs, double factor)
{
    matrix m(lhs.shape());
    const double* src_ptr = &lhs.get(0, 0);
    double* dst_ptr = &m.get(0, 0);
    for (std::size_t i = 0; i < m.size(); ++i) {
        *dst_ptr++ = *src_ptr++ * factor;
    }

    return m;
}

inline matrix& operator *= (matrix& lhs, double factor)
{
    double* ptr = &lhs.get(0, 0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        *ptr++ *= factor;
    }

    return lhs;
}

inline matrix operator + (const matrix& lhs, double value)
{
    matrix m(lhs.shape());
    const double* src_ptr = &lhs.get(0, 0);
    double* dst_ptr = &m.get(0, 0);
    for (std::size_t i = 0; i < m.size(); ++i) {
        *dst_ptr++ = *src_ptr++ + value;
    }

    return m;
}

inline matrix& operator += (matrix& lhs, double value)
{
    double* ptr = &lhs.get(0, 0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        *ptr++ += value;
    }

    return lhs;
}

inline matrix operator - (const matrix& lhs, double value)
{
    matrix m(lhs.shape());
    const double* src_ptr = &lhs.get(0, 0);
    double* dst_ptr = &m.get(0, 0);
    for (std::size_t i = 0; i < m.size(); ++i) {
        *dst_ptr++ = *src_ptr++ - value;
    }

    return m;
}

inline matrix& operator -= (matrix& lhs, double value)
{
    double* ptr = &lhs.get(0, 0);
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        *ptr++ -= value;
    }

    return lhs;
}

}