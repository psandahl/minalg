#include <minalg/linear.hpp>
#include <minalg/matrix.hpp>

#include <gtest/gtest.h>

#include <iostream>

TEST(LinearTest, Solve1)
{
    const minalg::Matrix A(minalg::Matrix({1, 3, -1, 4, -1, 1, 2, 4, 3}).reshape({3, 3}));
    const minalg::Matrix b({13, 9, -6});
    
    const minalg::Matrix x(minalg::solve(A, b));
    EXPECT_EQ(x.rows(), 3);
    EXPECT_EQ(x.columns(), 1);
    EXPECT_DOUBLE_EQ(x.at(0, 0), 4.0);
    EXPECT_DOUBLE_EQ(x.at(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(x.at(2, 0), -6.0);
}