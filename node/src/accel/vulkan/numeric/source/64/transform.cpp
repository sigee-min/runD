#include "../../transform.hpp"
#include "../../source.hpp"

#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] std::string TransformSource64() {
  return TransformProgramSource(NumericBaseSource64(), R"GLSL(
#define RUND_TRANSFORM_SCALAR int64_t
#define RUND_TRANSFORM_INDEX(value) ix64(value)
#define RUND_TRANSFORM_ADD add_q63
#define RUND_TRANSFORM_SUB sub_q63
#define RUND_TRANSFORM_MUL mul_q63
#define RUND_TRANSFORM_DIV(value, divisor) ((value) / int64_t(divisor))
)GLSL");
}

} // namespace rund::node::accel::detail
