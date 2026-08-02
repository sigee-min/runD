#include "../transform.hpp"
#include "../source.hpp"

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitTransformSource(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return EmitNumericBaseSource(sink, false) &&
         AppendTransformProgramSource(sink, R"GLSL(
#define RUND_TRANSFORM_SCALAR int
#define RUND_TRANSFORM_INDEX(value) uint(value)
#define RUND_TRANSFORM_ADD add_q31
#define RUND_TRANSFORM_SUB sub_q31
#define RUND_TRANSFORM_MUL mul_q31
#define RUND_TRANSFORM_DIV(value, divisor) int((value) / int(divisor))
)GLSL");
}

} // namespace

[[nodiscard]] std::string TransformSource() {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [](auto &sink)
      noexcept(noexcept(EmitTransformSource(sink))) {
    return EmitTransformSource(sink);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool TransformSourceBytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitTransformSource(sink);
      },
      bytes);
}

} // namespace rund::node::accel::detail
