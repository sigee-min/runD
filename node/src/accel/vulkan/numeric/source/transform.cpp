#include "../transform.hpp"
#include "../source.hpp"

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string TransformSource() {
  return TransformProgramSource(NumericBaseSource(), R"GLSL(
#define RUND_TRANSFORM_SCALAR int
#define RUND_TRANSFORM_INDEX(value) uint(value)
#define RUND_TRANSFORM_ADD add_q31
#define RUND_TRANSFORM_SUB sub_q31
#define RUND_TRANSFORM_MUL mul_q31
#define RUND_TRANSFORM_DIV(value, divisor) int((value) / int(divisor))
)GLSL");
}

} // namespace rund::node::accel::detail
