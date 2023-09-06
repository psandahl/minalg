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