#include "../../matrix.hpp"
#include "../../source.hpp"

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string MatrixSource64() {
  return MatrixProgramSource(NumericBaseSource64(), R"GLSL(
#define RUND_MATRIX_SCALAR int64_t
#define RUND_MATRIX_INDEX(value) ix64(value)
#define RUND_MATRIX_MIDX midx64
#define RUND_MATRIX_ZERO int64_t(0)
#define RUND_MATRIX_ADD matrix_add_i64
#define RUND_MATRIX_MUL matrix_mul_i64
)GLSL");
}

} // namespace rund::node::accel::detail
