#include <minalg/linear.hpp>
#include <minalg/matrix.hpp>

#include <gtest/gtest.h>

TEST(LinearTest, Solve1)
{
    const minalg::Matrix A(minalg::Matrix({-3, 2, -1, 6, -6, 7, 3, -4, 4}).reshape({3, 3}));
    const minalg::Matrix b({-1, -7, -6});
    
    const minalg::Matrix x(minalg::solve(A, b));
    EXPECT_EQ(x.rows(), 3);
    EXPECT_EQ(x.columns(), 1);
    EXPECT_DOUBLE_EQ(x.at(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(x.at(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(x.at(2, 0), -1.0);
}