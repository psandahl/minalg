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

}