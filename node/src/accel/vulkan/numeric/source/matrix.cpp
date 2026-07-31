#include "../matrix.hpp"
#include "../source.hpp"

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string MatrixSource() {
  return MatrixProgramSource(NumericBaseSource(), R"GLSL(
#define RUND_MATRIX_SCALAR int
#define RUND_MATRIX_INDEX(value) uint(value)
#define RUND_MATRIX_MIDX midx
#define RUND_MATRIX_ZERO 0
#define RUND_MATRIX_ADD matrix_add_i32
#define RUND_MATRIX_MUL matrix_mul_i32
)GLSL");
}

} // namespace rund::node::accel::detail
