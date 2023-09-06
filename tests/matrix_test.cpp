#include "minalg/matrix.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

TEST(MatrixTest, ValidConstruction)
{
    static constexpr std::size_t rows = 6;
    static constexpr std::size_t columns = 3;

    minalg::Matrix m(rows, columns);
    EXPECT_EQ(m.rows(), rows);
    EXPECT_EQ(m.columns(), columns);
    EXPECT_EQ(m.size(), rows * columns);

    const auto [num_rows, num_columns] = m.shape();
    EXPECT_EQ(num_rows, rows);
    EXPECT_EQ(num_columns, columns);
}

TEST(MatrixTest, InvalidConstruction) 
{
    EXPECT_THROW(minalg::Matrix(0, 0), std::invalid_argument);  
    EXPECT_THROW(minalg::Matrix(1, 0), std::invalid_argument);  
    EXPECT_THROW(minalg::Matrix(0, 1), std::invalid_argument);      
}

TEST(MatrixTest, LiteralConstruction)
{
    const std::vector<double> vec = { 1.0, 2.0, 3.0, 4.0 };
    
    minalg::Matrix m(vec);
    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.columns(), vec.size());
    EXPECT_DOUBLE_EQ(m.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m.get(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m.get(0, 3), 4.0);
}

TEST(MatrixTest, CopyConstruction)
{
    static constexpr std::size_t dim = 2;

    minalg::Matrix m0(dim, dim);
    m0.at(0, 0) = 1.0;
    m0.at(0, 1) = 2.0;
    m0.at(1, 0) = 3.0;
    m0.at(1, 1) = 4.0;

    minalg::Matrix m1(m0);
    EXPECT_EQ(m0.rows(), dim);
    EXPECT_EQ(m1.rows(), dim);
    EXPECT_EQ(m0.columns(), dim);
    EXPECT_EQ(m1.columns(), dim);
    EXPECT_DOUBLE_EQ(m1.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.get(1, 1), 4.0);
}

TEST(MatrixTest, CopyAssignment)
{
    static constexpr std::size_t dim = 2;

    minalg::Matrix m0(dim, dim);
    m0.at(0, 0) = 1.0;
    m0.at(0, 1) = 2.0;
    m0.at(1, 0) = 3.0;
    m0.at(1, 1) = 4.0;

    minalg::Matrix m1(3, 6);
    m1 = m0;
    EXPECT_EQ(m0.rows(), dim);
    EXPECT_EQ(m1.rows(), dim);
    EXPECT_EQ(m0.columns(), dim);
    EXPECT_EQ(m1.columns(), dim);
    EXPECT_DOUBLE_EQ(m1.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.get(1, 1), 4.0);
}

TEST(MatrixTest, MoveConstruction)
{
    static constexpr std::size_t dim = 2;

    minalg::Matrix m0(2, 2);
    m0.at(0, 0) = 1.0;
    m0.at(0, 1) = 2.0;
    m0.at(1, 0) = 3.0;
    m0.at(1, 1) = 4.0;

    minalg::Matrix m1(std::move(m0));
    EXPECT_EQ(m0.rows(), 0);
    EXPECT_EQ(m1.rows(), dim);
    EXPECT_EQ(m0.columns(), 0);
    EXPECT_EQ(m1.columns(), dim);
    EXPECT_DOUBLE_EQ(m1.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.get(1, 1), 4.0);    
}

TEST(MatrixTest, MoveAssignment)
{
    static constexpr std::size_t dim = 2;

    minalg::Matrix m0(2, 2);
    m0.at(0, 0) = 1.0;
    m0.at(0, 1) = 2.0;
    m0.at(1, 0) = 3.0;
    m0.at(1, 1) = 4.0;

    minalg::Matrix m1(3, 6);
    m1 = std::move(m0);
    EXPECT_EQ(m0.rows(), 0);
    EXPECT_EQ(m1.rows(), dim);
    EXPECT_EQ(m0.columns(), 0);
    EXPECT_EQ(m1.columns(), dim);
    EXPECT_DOUBLE_EQ(m1.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.get(1, 1), 4.0); 
}

TEST(MatrixTest, SetAndGet)
{
    minalg::Matrix m(2, 2);
    m.at(0, 0) = 1.0;
    m.at(0, 1) = 2.0;
    m.at(1, 0) = 3.0;
    m.at(1, 1) = 4.0;

    EXPECT_DOUBLE_EQ(m.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.get(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m.get(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m.get(1, 1), 4.0);
}

TEST(MatrixTest, AccessOutOfRange)
{
    minalg::Matrix m(3, 3);
    EXPECT_THROW(m.at(2, 3), std::out_of_range);
    EXPECT_THROW(m.at(3, 2), std::out_of_range);
    EXPECT_THROW(m.at(3, 3), std::out_of_range);
}

TEST(MatrixTest, Eye)
{
    minalg::Matrix m(minalg::Matrix::eye(2));
    EXPECT_DOUBLE_EQ(m.get(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.get(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(m.get(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(m.get(1, 1), 1.0);
}