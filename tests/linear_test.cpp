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
    EXPECT_THROW(minalg::linear::solve(A, b), minalg::linear::singular_matrix);
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
    EXPECT_THROW(minalg::linear::invert(A), minalg::linear::singular_matrix);    
}

TEST(LinearTest, LuDecomp1)
{
    const std::vector<double> data =
    {
        1, 2, 3, 
        0, 1, 4,
        5, 6, 0,        
    };
    const minalg::matrix A(minalg::matrix(data).reshape({3, 3}));
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

TEST(LinearTest, LuDecomp2)
{    
    const std::vector<double> data =
    {
        -2, 2, -1, 
        6, -6, 7,
        3, -8, 4,        
    };

    const minalg::matrix A(minalg::matrix(data).reshape({3, 3}));
    const auto [P, L, U] = minalg::linear::lu_decomp(A);

    // Reconstruct A from P, L and U.
    const minalg::matrix AA(P * L * U);
    EXPECT_TRUE(A == AA);
}

TEST(LinearTest, LuDecomp3)
{    
    const std::vector<double> data =
    {
        3, 1, -2, 5, -1,
        2, 7, -8, 9, 3,
        1, -4, 2, 6, 1,
        1, 5, 0, -2, 2,
        -1, 0, 3, -3, 6,
    };

    const minalg::matrix A(minalg::matrix(data).reshape({5, 5}));
    const auto [P, L, U] = minalg::linear::lu_decomp(A);

    // Reconstruct A from P, L and U.
    const minalg::matrix AA(P * L * U);
    EXPECT_TRUE(A == AA);
}

TEST(LinearTest, Det)
{
    const minalg::matrix m0(minalg::matrix({1, 0, 0, 0, 1, 0, 0, 0, 1}).reshape({3, 3}));
    EXPECT_DOUBLE_EQ(minalg::linear::det(m0), 1.0);

    const minalg::matrix m1(minalg::matrix({0, 0, 1, 0, 1, 0, 1, 0, 0}).reshape({3, 3}));
    EXPECT_DOUBLE_EQ(minalg::linear::det(m1), -1.0);

    const minalg::matrix m2(minalg::matrix({0, 1, 0, 0, 0, 1, 1, 0, 0}).reshape({3, 3}));
    EXPECT_DOUBLE_EQ(minalg::linear::det(m2), 1.0);

    // Compared with numpy.
    const std::vector<double> data3 =
    {
        3, 1, -2, 5, -1,
        2, 7, -8, 9, 3,
        0, -4, 2, 6, 1,
        1, 5, 0, -2, 2,
        -1, 0, 3, -3, 6,
    };
    const minalg::matrix m3(minalg::matrix(data3).reshape({5, 5}));
    EXPECT_DOUBLE_EQ(minalg::linear::det(m3), -3157.0);

    const std::vector<double> data4 =
    {
        -1, 0, 3, -3, 6,        
        2, 7, -8, 9, 3,
        0, -4, 2, 6, 1,
        1, 5, 0, -2, 2,        
        3, 1, -2, 5, -1,
    };
    const minalg::matrix m4(minalg::matrix(data4).reshape({5, 5}));
    EXPECT_DOUBLE_EQ(minalg::linear::det(m4), -3157.0);

    const minalg::matrix m5(minalg::matrix({0, 1, 0, 0, 0, 1, 0, 0, 1}).reshape({3, 3}));
    EXPECT_DOUBLE_EQ(minalg::linear::det(m5), 0.0);
}

TEST(LinearTest, QrDecomp1)
{
    minalg::matrix A({1, 2, 3, 4, 5, 6});
    A.reshaped({3, 2});

    const auto [Q, R] = minalg::linear::qr_decomp(A);
    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.columns(), 3);
    EXPECT_EQ(R.rows(), 3);
    EXPECT_EQ(R.columns(), 2);

    // Reconstruct from Q * R.
    const minalg::matrix AA(Q * R);
    EXPECT_TRUE(A == AA);

    // Q shall be orthogonal.
    EXPECT_TRUE(Q.is_orthogonal());

    // Specific values for Q and R. Taken from Octave, as the implementation
    // is inspired by a Matlab example.
    EXPECT_NEAR(Q.at(0, 0),  -0.16903, 1e-05);
    EXPECT_NEAR(Q.at(0, 1),  0.89709, 1e-05);
    EXPECT_NEAR(Q.at(0, 2),  0.40825, 1e-05);

    EXPECT_NEAR(Q.at(1, 0),  -0.50709, 1e-05);
    EXPECT_NEAR(Q.at(1, 1),  0.27603, 1e-05);
    EXPECT_NEAR(Q.at(1, 2),  -0.81650, 1e-05);

    EXPECT_NEAR(Q.at(2, 0),  -0.84515, 1e-05);
    EXPECT_NEAR(Q.at(2, 1),  -0.34503, 1e-05);
    EXPECT_NEAR(Q.at(2, 2),  0.40825, 1e-05);

    EXPECT_NEAR(R.at(0, 0), -5.91608, 1e-05);
    EXPECT_NEAR(R.at(0, 1), -7.43736, 1e-05);

    EXPECT_DOUBLE_EQ(R.at(1, 0), 0.0);
    EXPECT_NEAR(R.at(1, 1), 0.82808, 1e-05);

    EXPECT_DOUBLE_EQ(R.at(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(R.at(2, 1), 0.0);
}

TEST(LinearTest, QrDecomp2)
{
    minalg::matrix A({1, 2, 3, 4, 5, 6, 7, 8, 9});
    A.reshaped({3, 3});

    const auto [Q, R] = minalg::linear::qr_decomp(A);
    EXPECT_EQ(Q.rows(), 3);
    EXPECT_EQ(Q.columns(), 3);
    EXPECT_EQ(R.rows(), 3);
    EXPECT_EQ(R.columns(), 3);

    // Reconstruct from Q * R.
    const minalg::matrix AA(Q * R);
    EXPECT_TRUE(A == AA);

    // Q shall be orthogonal.
    EXPECT_TRUE(Q.is_orthogonal());

    // Specific values for Q and R. Taken from Octave, as the implementation
    // is inspired by a Matlab example.
    EXPECT_NEAR(Q.at(0, 0),  -0.12309, 1e-05);
    EXPECT_NEAR(Q.at(0, 1),  0.90453, 1e-05);
    EXPECT_NEAR(Q.at(0, 2),  0.40825, 1e-05);

    EXPECT_NEAR(Q.at(1, 0),  -0.49237, 1e-05);
    EXPECT_NEAR(Q.at(1, 1),  0.30151, 1e-05);
    EXPECT_NEAR(Q.at(1, 2),  -0.81650, 1e-05);

    EXPECT_NEAR(Q.at(2, 0),  -0.86164, 1e-05);
    EXPECT_NEAR(Q.at(2, 1),  -0.30151, 1e-05);
    EXPECT_NEAR(Q.at(2, 2),  0.40825, 1e-05);

    EXPECT_NEAR(R.at(0, 0),  -8.12404, 1e-05);
    EXPECT_NEAR(R.at(0, 1),  -9.60114, 1e-05);
    EXPECT_NEAR(R.at(0, 2),  -11.07823, 1e-05);

    EXPECT_DOUBLE_EQ(R.at(1, 0), 0.0);
    EXPECT_NEAR(R.at(1, 1),  0.90453, 1e-05);
    EXPECT_NEAR(R.at(1, 2),  1.80907, 1e-05);

    EXPECT_DOUBLE_EQ(R.at(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(R.at(2, 1), 0.0);
    EXPECT_NEAR(R.at(2, 2), -0.0, 1e-05);
}

TEST(LinearTest, InvalidQrDecomp)
{
    minalg::matrix A({1, 2, 3, 4, 5, 6});
    A.reshaped({2, 3});

    EXPECT_THROW(minalg::linear::qr_decomp(A), std::invalid_argument);
}
