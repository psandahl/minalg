#include <minalg/linear.hpp>
#include <minalg/matrix.hpp>

#include <vector>

#include <gtest/gtest.h>

#include <iostream>

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
    EXPECT_THROW(minalg::linear::solve(A, b), std::invalid_argument);
}

TEST(LinearTest, InvalidSolve2)
{
    const minalg::matrix A(minalg::matrix({1, 2, 3, 4, 5, 6}).reshape({3, 2}));
    const minalg::matrix b(std::vector<double>({1, 2, 3}));
    EXPECT_THROW(minalg::linear::solve(A, b), std::invalid_argument);
}

TEST(LinearTest, Invert1)
{
    const minalg::matrix A(minalg::matrix({1, 2, 3, 0, 1, 4, 5, 6, 0}).reshape({3, 3}));

    const minalg::matrix Ainv(minalg::linear::invert(A));
    EXPECT_EQ(Ainv.rows(), A.rows());
    EXPECT_EQ(Ainv.columns(), A.columns());

    // Hardcoded values.
    EXPECT_NEAR(Ainv.at(0, 0), -24, 1e-09);
    EXPECT_NEAR(Ainv.at(0, 1), 18, 1e-09);
    EXPECT_NEAR(Ainv.at(0, 2), 5, 1e-09);

    EXPECT_NEAR(Ainv.at(1, 0), 20, 1e-09);
    EXPECT_NEAR(Ainv.at(1, 1), -15, 1e-09);
    EXPECT_NEAR(Ainv.at(1, 2), -4, 1e-09);

    EXPECT_NEAR(Ainv.at(2, 0), -5, 1e-09);
    EXPECT_NEAR(Ainv.at(2, 1), 4, 1e-09);
    EXPECT_NEAR(Ainv.at(2, 2), 1, 1e-09);

    // Check Ainv * A == I.
    EXPECT_TRUE(Ainv * A == minalg::matrix::eye(A.rows()));
}

TEST(LinearTest, Invert2)
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
    const minalg::matrix Ainv(minalg::linear::invert(A));
    EXPECT_EQ(Ainv.rows(), A.rows());
    EXPECT_EQ(Ainv.columns(), A.columns());

    // Check Ainv * A == I.
    EXPECT_TRUE(Ainv * A == minalg::matrix::eye(A.rows()));

    // Solve x = Ainv * b (same as in Solve2).
    const minalg::matrix b({-16, -55, 14, 29, 67});
    const minalg::matrix x(Ainv * b);

    EXPECT_EQ(x.rows(), 5);
    EXPECT_EQ(x.columns(), 1);
    EXPECT_DOUBLE_EQ(x.at(0, 0), 5.0);
    //EXPECT_DOUBLE_EQ(x.at(1, 0), 2.0);
    EXPECT_NEAR(x.at(1, 0), 2.0, 1e-14);
    EXPECT_DOUBLE_EQ(x.at(2, 0), 11.0);
    //EXPECT_DOUBLE_EQ(x.at(3, 0), -1.0);
    EXPECT_NEAR(x.at(3, 0), -1.0, 1e-14);
    EXPECT_DOUBLE_EQ(x.at(4, 0), 6.0);
}

TEST(LinearTest, InvalidInvert)
{
    const minalg::matrix A(minalg::matrix({0, 1, 2, 3, 4, 5, 6, 7, 8}).reshape({3, 3}));
    EXPECT_THROW(minalg::linear::invert(A), std::invalid_argument);
}

TEST(LinearTest, lu_decomp1)
{
    const minalg::matrix A(minalg::matrix({1, 2, 3, 0, 1, 4, 5, 6, 0}).reshape({3, 3}));
    const auto [P, L, U] = minalg::linear::lu_decomp(A);

    // Reconstruct A from P, L and U.
    const minalg::matrix AA(P * L * U);
    EXPECT_TRUE(A == AA);

    // Compare with values from scipy lu decomp.
    EXPECT_DOUBLE_EQ(P.at(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(P.at(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(P.at(0, 2), 1.0);

    EXPECT_DOUBLE_EQ(P.at(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(P.at(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(P.at(1, 2), 0.0);

    EXPECT_DOUBLE_EQ(P.at(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(P.at(2, 1), 0.0);
    EXPECT_DOUBLE_EQ(P.at(2, 2), 0.0);

    EXPECT_DOUBLE_EQ(L.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(L.at(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(L.at(0, 2), 0.0);

    EXPECT_DOUBLE_EQ(L.at(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(L.at(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(L.at(1, 2), 0.0);

    EXPECT_DOUBLE_EQ(L.at(2, 0), 0.2);
    EXPECT_DOUBLE_EQ(L.at(2, 1), 0.8);
    EXPECT_DOUBLE_EQ(L.at(2, 2), 1.0);

    EXPECT_DOUBLE_EQ(U.at(0, 0), 5.0);
    EXPECT_DOUBLE_EQ(U.at(0, 1), 6.0);
    EXPECT_DOUBLE_EQ(U.at(0, 2), 0.0);

    EXPECT_DOUBLE_EQ(U.at(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(U.at(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(U.at(1, 2), 4.0);

    EXPECT_DOUBLE_EQ(U.at(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(U.at(2, 1), 0.0);
    EXPECT_NEAR(U.at(2, 2), -0.2, 1e-15);
}

TEST(LinearTest, lu_decomp2)
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
    const auto [P, L, U] = minalg::linear::lu_decomp(A);

    // Reconstruct A from P, L and U.
    const minalg::matrix AA(P * L * U);

    std::cout << "A=\n" << A.info() << std::endl;    
    std::cout << "AA=\n" << AA.info() << std::endl;
    std::cout << "P=\n" << P.info() << std::endl;    
    std::cout << "L=\n" << L.info() << std::endl;    
    std::cout << "U=\n" << U.info() << std::endl;    

    //EXPECT_TRUE(A == AA);
#if 0
    >>> P
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 1.]])
>>> L
array([[ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.66666667,  1.        ,  0.        ,  0.        ,  0.        ],
       [ 0.33333333,  0.73684211,  1.        ,  0.        ,  0.        ],
       [ 0.        , -0.63157895, -0.39622642,  1.        ,  0.        ],
       [-0.33333333,  0.05263158,  0.48113208,  0.33090379,  1.        ]])
>>> U
array([[ 3.        ,  1.        , -2.        ,  5.        , -1.        ],
       [ 0.        ,  6.33333333, -6.66666667,  5.66666667,  3.66666667],
       [ 0.        ,  0.        ,  5.57894737, -7.84210526, -0.36842105],
       [ 0.        ,  0.        ,  0.        ,  6.47169811,  3.16981132],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  4.60204082]])
#endif
}
