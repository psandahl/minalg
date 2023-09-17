#include <minalg/linear.hpp>
#include <minalg/matrix.hpp>

#include <vector>

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

    // Also check that Ax = b.
    const minalg::Matrix b2(minalg::Matrix::multiply(A, x));
    EXPECT_EQ(b2.rows(), b.rows());
    EXPECT_EQ(b2.columns(), b.columns());
    EXPECT_DOUBLE_EQ(b2.at(0, 0), b.at(0, 0));
    EXPECT_DOUBLE_EQ(b2.at(1, 0), b.at(1, 0));
    EXPECT_DOUBLE_EQ(b2.at(2, 0), b.at(2, 0));
}

TEST(LinearTest, Solve2)
{
    const std::vector<double> data =
    {
        3, 1, -2, 5, -1,
        2, 7, -8, 9, 3,
        0, -4, 2, 6, 1,
        1, 5, 0, -2, 2,
        -1, 0, 3, -3, 6
    };

    const minalg::Matrix A(minalg::Matrix(data).reshape({5, 5}));
    const minalg::Matrix b({-16, -55, 14, 29, 67});
    const minalg::Matrix x(minalg::solve(A, b));
    EXPECT_EQ(x.rows(), 5);
    EXPECT_EQ(x.columns(), 1);
    EXPECT_DOUBLE_EQ(x.at(0, 0), 5.0);
    //EXPECT_DOUBLE_EQ(x.at(1, 0), 2.0);
    EXPECT_NEAR(x.at(1, 0), 2.0, 1e-14);
    EXPECT_DOUBLE_EQ(x.at(2, 0), 11.0);
    //EXPECT_DOUBLE_EQ(x.at(3, 0), -1.0);
    EXPECT_NEAR(x.at(3, 0), -1.0, 1e-14);
    EXPECT_DOUBLE_EQ(x.at(4, 0), 6.0);

    // Also check that Ax = b.
    const minalg::Matrix b2(minalg::Matrix::multiply(A, x));
    EXPECT_EQ(b2.rows(), b.rows());
    EXPECT_EQ(b2.columns(), b.columns());
    EXPECT_DOUBLE_EQ(b2.at(0, 0), b.at(0, 0));
    EXPECT_DOUBLE_EQ(b2.at(1, 0), b.at(1, 0));
    EXPECT_DOUBLE_EQ(b2.at(2, 0), b.at(2, 0));
    EXPECT_DOUBLE_EQ(b2.at(3, 0), b.at(3, 0));
    EXPECT_DOUBLE_EQ(b2.at(4, 0), b.at(4, 0));
}
