#include <minalg/matrix.hpp>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

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
    EXPECT_EQ(m.rows(), vec.size());
    EXPECT_EQ(m.columns(), 1);
    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(m.at(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(m.at(3, 0), 4.0);
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
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 4.0);
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
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 4.0);
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
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 4.0);    
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
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 4.0); 
}

TEST(MatrixTest, SetAndGet)
{
    minalg::Matrix m(2, 2);
    m.at(0, 0) = 1.0;
    m.at(0, 1) = 2.0;
    m.at(1, 0) = 3.0;
    m.at(1, 1) = 4.0;

    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m.at(1, 1), 4.0);
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
    EXPECT_DOUBLE_EQ(m.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m.at(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(m.at(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(m.at(1, 1), 1.0);
}

TEST(MatrixTest, ValidDiag)
{
    const std::vector<double> vec0 = {1.0, 2.0, 3.0};
    minalg::Matrix m(minalg::Matrix::diag(vec0));
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

TEST(MatrixTest, InvalidDiag)
{
    minalg::Matrix m(3, 2);
    EXPECT_THROW(m.diag(), std::invalid_argument);
}

TEST(MatrixTest, ValidReshape)
{
    const std::vector<double> vec = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    minalg::Matrix m0(vec);

    minalg::Matrix m1(m0.reshape({2, 3}));
    EXPECT_EQ(m1.rows(), 2);
    EXPECT_EQ(m1.columns(), 3);
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 4.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 2), 6.0);
}

TEST(MatrixTest, InvalidReshape)
{
    const std::vector<double> vec = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    minalg::Matrix m0(vec);

    EXPECT_THROW(m0.reshape({2, 2}), std::invalid_argument);
}

TEST(MatrixTest, ValidTranspose)
{
    minalg::Matrix m0({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    m0.reshaped({2, 3});

    minalg::Matrix m1(m0.transpose());
    EXPECT_EQ(m1.rows(), m0.columns());
    EXPECT_EQ(m1.columns(), m0.rows());
    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 4.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(m1.at(2, 0), 3.0);
    EXPECT_DOUBLE_EQ(m1.at(2, 1), 6.0);
}

TEST(MatrixTest, InvalidTranspose)
{
    minalg::Matrix m0(3, 3);
    minalg::Matrix m1(4, 4);

    EXPECT_THROW(minalg::Matrix::transpose(m0, m1), std::invalid_argument);
}

TEST(MatrixTest, DotProduct)
{
    minalg::Matrix m({1.0, 2.0, 3.0}); // Column matrix.
    minalg::Matrix m1(minalg::Matrix::multiply(m.transpose(), m));

    EXPECT_EQ(m1.rows(), 1);
    EXPECT_EQ(m1.columns(), 1);

    const double ip = 1. * 1. + 2. * 2. + 3. * 3.;
    EXPECT_DOUBLE_EQ(m1.at(0, 0), ip);
}

TEST(MatrixTest, MulEye)
{
    minalg::Matrix m(minalg::Matrix::eye(5));
    minalg::Matrix m1(minalg::Matrix::multiply(m, m));

    EXPECT_EQ(m1.rows(), 5);
    EXPECT_EQ(m1.columns(), 5);

    std::vector<double> diag(m1.diag());
    EXPECT_DOUBLE_EQ(diag[0], 1.0);
    EXPECT_DOUBLE_EQ(diag[1], 1.0);
    EXPECT_DOUBLE_EQ(diag[2], 1.0);
    EXPECT_DOUBLE_EQ(diag[3], 1.0);
    EXPECT_DOUBLE_EQ(diag[4], 1.0);
}

TEST(MatrixTest, MatSqr)
{
    minalg::Matrix m({1.0, 2.0, 3.0, 4.0});
    m.reshaped({2, 2});

    minalg::Matrix m1(minalg::Matrix::multiply(m, m));

    EXPECT_EQ(m1.rows(), 2);
    EXPECT_EQ(m1.columns(), 2);

    EXPECT_DOUBLE_EQ(m1.at(0, 0), 1. * 1. + 2. * 3.);
    EXPECT_DOUBLE_EQ(m1.at(0, 1), 1. * 2. + 2. * 4.);
    EXPECT_DOUBLE_EQ(m1.at(1, 0), 3. * 1. + 4. * 3.);
    EXPECT_DOUBLE_EQ(m1.at(1, 1), 3. * 2. + 4. * 4.);
}

TEST(MatrixTest, InvalidMultiplication1)
{
    minalg::Matrix m0(3, 4);
    minalg::Matrix m1(2, 1);
    minalg::Matrix m2(3, 1);

    EXPECT_THROW(minalg::Matrix::multiply(m0, m1, m2), std::invalid_argument);
}

TEST(MatrixTest, InvalidMultiplication2)
{
    minalg::Matrix m0(3, 4);
    minalg::Matrix m1(4, 1);
    minalg::Matrix m2(3, 2);

    EXPECT_THROW(minalg::Matrix::multiply(m0, m1, m2), std::invalid_argument);
}

TEST(MatrixTest, ValidHconcat)
{
    minalg::Matrix m0({1.0, 2.0, 3.0, 4.0});
    m0.reshaped({2, 2});

    minalg::Matrix m1(std::vector<double>({5.0, 6.0}));

    minalg::Matrix m2(minalg::Matrix::hconcat(m0, m1));
    EXPECT_EQ(m2.rows(), m0.rows());
    EXPECT_EQ(m2.columns(), m0.columns() + m1.columns());

    EXPECT_DOUBLE_EQ(m2.at(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(m2.at(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m2.at(0, 2), 5.0);
    EXPECT_DOUBLE_EQ(m2.at(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(m2.at(1, 1), 4.0);
    EXPECT_DOUBLE_EQ(m2.at(1, 2), 6.0);
}

TEST(MatrixTest, InvalidHconcat1)
{
    minalg::Matrix m0(3, 3);
    minalg::Matrix m1(2, 3);
    minalg::Matrix m2(3, 3);

    EXPECT_THROW(minalg::Matrix::hconcat(m0, m1, m2), std::invalid_argument);
}

TEST(MatrixTest, InvalidHconcat2)
{
    minalg::Matrix m0(3, 3);
    minalg::Matrix m1(3, 3);
    minalg::Matrix m2(2, 3);

    EXPECT_THROW(minalg::Matrix::hconcat(m0, m1, m2), std::invalid_argument);
}