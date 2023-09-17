#include <minalg/matrix.hpp>

#include "util.hpp"

namespace minalg {

double inner_product(const Matrix& m0, const Matrix& m1,
                     std::size_t row_m0, std::size_t col_m1, 
                     std::size_t len)
{    
    double sum = 0.0;
    for (std::size_t i = 0; i < len; ++i) {
        sum += m0.get(row_m0, i) * m1.get(i, col_m1);
    }

    return sum;
}

std::vector<std::size_t> index_vector(std::size_t indices)
{
    std::vector<std::size_t> vec(indices);
    for (std::size_t i = 0; i < indices; ++i) {
        vec[i] = i;
    }

    return vec;
}

std::size_t find_pivot_row_index(std::size_t diag, 
                                 const Matrix& m, 
                                 const std::vector<std::size_t>& rows)
{
    std::size_t max_row = diag;
    double max_val = 0.0;

    for (std::size_t row = diag; row < rows.size(); ++row) {
        const double val = std::fabs(m.at(rows[row], diag));
        if (val > max_val) {
            max_row = row;
            max_val = val;
        }
    }

    return max_row;
}

}