#include "minalg/matrix.hpp"

#include <cstddef>
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