#include <minalg/matrix.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

TEST(matrixTest, ValidConstruction)
{
    static constexpr std::size_t rows = 6;
    static constexpr std::size_t columns = 3;

    minalg::matrix m(rows, columns);
    EXPECT_EQ(m.rows(), rows);
    EXPECT_EQ(m.columns(), columns);
    EXPECT_EQ(m.size(), rows * columns);

    const auto [num_rows, num_columns] = m.shape();
    EXPECT_EQ(num_rows, rows);
    EXPECT_EQ(num_columns, columns);
}

TEST(matrixTest, InvalidConstruction) 
{
    EXPECT_THROW(minalg::matrix(0, 0), std::invalid_argument);  
    EXPECT_THROW(minalg::matrix(1, 0), std::invalid_argument);  
    EXPECT_THROW(minalg::matrix(0, 1), std::invalid_argument);      
}

TEST(matrixTest, LiteralConstruction)
{
    const std::vector<double> vec = { 1.0, 2.0, 3.0, 4.0 };
    
    minalg::matrix m(vec);
    EXPECT_EQ(m.rows(), vec.size());
    EXPECT_EQ(m.columns(), 1);
    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(m.at(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(m.at(3, 0), 4.0);
}

TEST(matrixTest, CopyConstruction)
{
    static constexpr std::size_t dim = 2;

    minalg::matrix m0(dim, dim);
    m0.at(0, 0) = 1.0;
    m0.at(0, 1) = 2.0;
    m0.at(1, 0) = 3.0;
    m0.at(1, 1) = 4.0;

    minalg::matrix m1(m0);
    EXPECT_EQ(m0.rows(), dim);
    EXPECT_EQ(m1.rows(), dim);
    EXPECT_EQ(m0.columns(), dim);
    EXPECT_EQ(m1.columns(), dim);
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 4.0);
}

TEST(matrixTest, CopyAssignment)
{
    static constexpr std::size_t dim = 2;

    minalg::matrix m0(dim, dim);
    m0.at(0, 0) = 1.0;
    m0.at(0, 1) = 2.0;
    m0.at(1, 0) = 3.0;
    m0.at(1, 1) = 4.0;

    minalg::matrix m1(3, 6);
    m1 = m0;
    EXPECT_EQ(m0.rows(), dim);
    EXPECT_EQ(m1.rows(), dim);
    EXPECT_EQ(m0.columns(), dim);
    EXPECT_EQ(m1.columns(), dim);
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 4.0);
}

TEST(matrixTest, MoveConstruction)
{
    static constexpr std::size_t dim = 2;

    minalg::matrix m0(2, 2);
    m0.at(0, 0) = 1.0;
    m0.at(0, 1) = 2.0;
    m0.at(1, 0) = 3.0;
    m0.at(1, 1) = 4.0;

    minalg::matrix m1(std::move(m0));
    EXPECT_EQ(m0.rows(), 0);
    EXPECT_EQ(m1.rows(), dim);
    EXPECT_EQ(m0.columns(), 0);
    EXPECT_EQ(m1.columns(), dim);
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 4.0);    
}

TEST(matrixTest, MoveAssignment)
{
    static constexpr std::size_t dim = 2;

    minalg::matrix m0(2, 2);
    m0.at(0, 0) = 1.0;
    m0.at(0, 1) = 2.0;
    m0.at(1, 0) = 3.0;
    m0.at(1, 1) = 4.0;

    minalg::matrix m1(3, 6);
    m1 = std::move(m0);
    EXPECT_EQ(m0.rows(), 0);
    EXPECT_EQ(m1.rows(), dim);
    EXPECT_EQ(m0.columns(), 0);
    EXPECT_EQ(m1.columns(), dim);
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 4.0); 
}

TEST(matrixTest, SetAndGet)
{
    minalg::matrix m(2, 2);
    m.at(0, 0) = 1.0;
    m.at(0, 1) = 2.0;
    m.at(1, 0) = 3.0;
    m.at(1, 1) = 4.0;

    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m.at(1, 1), 4.0);
}

TEST(matrixTest, AccessOutOfRange)
{
    minalg::matrix m(3, 3);
    EXPECT_THROW(m.at(2, 3), std::out_of_range);
    EXPECT_THROW(m.at(3, 2), std::out_of_range);
    EXPECT_THROW(m.at(3, 3), std::out_of_range);
}

TEST(matrixTest, Eye)
{
    minalg::matrix m(minalg::matrix::eye(2));
    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(m.at(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(m.at(1, 1), 1.0);
}

TEST(matrixTest, ValidDiag)
{
    const std::vector<double> vec0 = {1.0, 2.0, 3.0};
    minalg::matrix m(minalg::matrix::diag(vec0));
    EXPECT_EQ(m.rows(), vec0.size());
    EXPECT_EQ(m.columns(), vec0.size());
    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(m.at(2, 2), 3.0);

    const std::vector<double> vec1(m.diag());
    EXPECT_EQ(vec0.size(), vec1.size());
    EXPECT_DOUBLE_EQ(vec0[0], vec1[0]);
    EXPECT_DOUBLE_EQ(vec0[1], vec1[1]);
    EXPECT_DOUBLE_EQ(vec0[2], vec1[2]);
}

TEST(matrixTest, InvalidDiag)
{
    minalg::matrix m(3, 2);
    EXPECT_THROW(m.diag(), std::invalid_argument);
}

TEST(matrixTest, ValidReshape)
{
    const std::vector<double> vec = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    minalg::matrix m0(vec);

    minalg::matrix m1(m0.reshape({2, 3}));
    EXPECT_EQ(m1.rows(), 2);
    EXPECT_EQ(m1.columns(), 3);
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 2), 6.0);
}

TEST(matrixTest, InvalidReshape)
{
    const std::vector<double> vec = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    minalg::matrix m0(vec);

    EXPECT_THROW(m0.reshape({2, 2}), std::invalid_argument);
}

TEST(matrixTest, ValidTranspose)
{
    minalg::matrix m0({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    m0.reshaped({2, 3});

    minalg::matrix m1(m0.transpose());
    EXPECT_EQ(m1.rows(), m0.columns());
    EXPECT_EQ(m1.columns(), m0.rows());
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(m1.at(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(2, 1), 6.0);
}

TEST(matrixTest, InvalidTranspose)
{
    minalg::matrix m0(3, 3);
    minalg::matrix m1(4, 4);

    EXPECT_THROW(minalg::matrix::transpose(m0, m1), std::invalid_argument);
}

TEST(matrixTest, DotProduct)
{
    minalg::matrix m({1.0, 2.0, 3.0}); // Column matrix.
    minalg::matrix m1(minalg::matrix::multiply(m.transpose(), m));

    EXPECT_EQ(m1.rows(), 1);
    EXPECT_EQ(m1.columns(), 1);

    const double ip = 1. * 1. + 2. * 2. + 3. * 3.;
    EXPECT_DOUBLE_EQ(m1.at(0, 0), ip);
}

TEST(matrixTest, MulEye)
{
    minalg::matrix m(minalg::matrix::eye(5));
    minalg::matrix m1(minalg::matrix::multiply(m, m));

    EXPECT_EQ(m1.rows(), 5);
    EXPECT_EQ(m1.columns(), 5);

    std::vector<double> diag(m1.diag());
    EXPECT_DOUBLE_EQ(diag[0], 1.0);
    EXPECT_DOUBLE_EQ(diag[1], 1.0);
    EXPECT_DOUBLE_EQ(diag[2], 1.0);
    EXPECT_DOUBLE_EQ(diag[3], 1.0);
    EXPECT_DOUBLE_EQ(diag[4], 1.0);
}

TEST(matrixTest, MatSqr)
{
    minalg::matrix m({1.0, 2.0, 3.0, 4.0});
    m.reshaped({2, 2});

    minalg::matrix m1(minalg::matrix::multiply(m, m));

    EXPECT_EQ(m1.rows(), 2);
    EXPECT_EQ(m1.columns(), 2);

    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1. * 1. + 2. * 3.);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 1. * 2. + 2. * 4.);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3. * 1. + 4. * 3.);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 3. * 2. + 4. * 4.);
}

TEST(matrixTest, InvalidMultiplication1)
{
    minalg::matrix m0(3, 4);
    minalg::matrix m1(2, 1);
    minalg::matrix m2(3, 1);

    EXPECT_THROW(minalg::matrix::multiply(m0, m1, m2), std::invalid_argument);
}

TEST(matrixTest, InvalidMultiplication2)
{
    minalg::matrix m0(3, 4);
    minalg::matrix m1(4, 1);
    minalg::matrix m2(3, 2);

    EXPECT_THROW(minalg::matrix::multiply(m0, m1, m2), std::invalid_argument);
}

TEST(matrixTest, ValidHconcat)
{
    minalg::matrix m0({1.0, 2.0, 3.0, 4.0});
    m0.reshaped({2, 2});

    minalg::matrix m1(std::vector<double>({5.0, 6.0}));

    minalg::matrix m2(minalg::matrix::hconcat(m0, m1));
    EXPECT_EQ(m2.rows(), m0.rows());
    EXPECT_EQ(m2.columns(), m0.columns() + m1.columns());

    EXPECT_DOUBLE_EQ(m2.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m2.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m2.at(0, 2), 5.0);
    EXPECT_DOUBLE_EQ(m2.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m2.at(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(m2.at(1, 2), 6.0);
}

TEST(matrixTest, InvalidHconcat1)
{
    minalg::matrix m0(3, 3);
    minalg::matrix m1(2, 3);
    minalg::matrix m2(3, 3);

    EXPECT_THROW(minalg::matrix::hconcat(m0, m1, m2), std::invalid_argument);
}

TEST(matrixTest, InvalidHconcat2)
{
    minalg::matrix m0(3, 3);
    minalg::matrix m1(3, 3);
    minalg::matrix m2(2, 3);

    EXPECT_THROW(minalg::matrix::hconcat(m0, m1, m2), std::invalid_argument);
}

TEST(matrixTest, Equal)
{
    minalg::matrix m0({1.0, 2.0, 3.0, 4.0});
    m0.reshaped({2, 2});
    EXPECT_TRUE(m0 == m0);

    minalg::matrix m1(m0);
    EXPECT_TRUE(m0 == m1);

    m1.at(1, 1) += 1e-10; // Difference within tolerances.
    EXPECT_TRUE(m0 == m1);

    minalg::matrix m2(m0);
    m2.at(1, 1) += 1e-6; // Difference not within tolerances.
    EXPECT_FALSE(m0 == m2);
}

TEST(matrixTest, IsSymmetric)
{
    minalg::matrix m0({1.0, 2.0, 2.0, 4.0});
    m0.reshaped({2, 2});

    EXPECT_TRUE(m0.is_symmetric());

    m0.at(1, 0) = 3.0;
    EXPECT_FALSE(m0.is_symmetric());
}

TEST(matrixTest, IsOrthogonal)
{
    const std::vector<double> data =
    {
        1, 0, 0,
        0, 0.70710678, -0.70710678,
        0, 0.70710678, 0.70710678
    };

    minalg::matrix m0(data);
    m0.reshaped({3, 3});

    EXPECT_TRUE(m0.is_orthogonal());    

    m0.at(0, 1) = 1e-03;
    EXPECT_FALSE(m0.is_orthogonal());    
}