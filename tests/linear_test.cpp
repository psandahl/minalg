#include <minalg/linear.hpp>
#include <minalg/matrix.hpp>

#include <vector>

#include <gtest/gtest.h>

TEST(LinearTest, Solve1)
{
    const minalg::matrix A(minalg::matrix({-3, 2, -1, 6, -6, 7, 3, -4, 4}).reshape({3, 3}));
    const minalg::matrix b({-1, -7, -6});
    
    const minalg::matrix x(minalg::linear::solve(A, b));
    EXPECT_EQ(x.rows(), 3);
    EXPECT_EQ(x.columns(), 1);
    EXPECT_DOUBLE_EQ(x.at(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(x.at(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(x.at(2, 0), -1.0);

    // Also check that Ax = b.
    const minalg::matrix b2(minalg::matrix::multiply(A, x));
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

    const minalg::matrix A(minalg::matrix(data).reshape({5, 5}));
    const minalg::matrix b({-16, -55, 14, 29, 67});
    const minalg::matrix x(minalg::linear::solve(A, b));
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
    const minalg::matrix b2(minalg::matrix::multiply(A, x));
    EXPECT_EQ(b2.rows(), b.rows());
    EXPECT_EQ(b2.columns(), b.columns());
    EXPECT_DOUBLE_EQ(b2.at(0, 0), b.at(0, 0));
    EXPECT_DOUBLE_EQ(b2.at(1, 0), b.at(1, 0));
    EXPECT_DOUBLE_EQ(b2.at(2, 0), b.at(2, 0));
    EXPECT_DOUBLE_EQ(b2.at(3, 0), b.at(3, 0));
    EXPECT_DOUBLE_EQ(b2.at(4, 0), b.at(4, 0));
}

TEST(LinearTest, InvalidSolve1)
{
    const minalg::matrix A(minalg::matrix({0, 1, 0, -1}).reshape({2, 2}));
    const minalg::matrix b(std::vector<double>({2, 3}));
    EXPECT_THROW(minalg::linear::solve(A, b), std::out_of_range);
}

TEST(LinearTest, InvalidSolve2)
{
    const minalg::matrix A(minalg::matrix({1, 2, 3, 4, 5, 6}).reshape({3, 2}));
    const minalg::matrix b(std::vector<double>({1, 2, 3}));
    EXPECT_THROW(minalg::linear::solve(A, b), std::invalid_argument);
}
