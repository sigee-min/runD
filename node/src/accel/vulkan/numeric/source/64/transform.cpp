#include "../../transform.hpp"
#include "../../source.hpp"

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitTransformSource64(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return EmitNumericBaseSource(sink, true) &&
         AppendTransformProgramSource(sink, R"GLSL(
#define RUND_TRANSFORM_SCALAR int64_t
#define RUND_TRANSFORM_INDEX(value) ix64(value)
#define RUND_TRANSFORM_ADD add_q63
#define RUND_TRANSFORM_SUB sub_q63
#define RUND_TRANSFORM_MUL mul_q63
#define RUND_TRANSFORM_DIV(value, divisor) ((value) / int64_t(divisor))
)GLSL");
}

} // namespace

[[nodiscard]] std::string TransformSource64() {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [](auto &sink)
      noexcept(noexcept(EmitTransformSource64(sink))) {
    return EmitTransformSource64(sink);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool TransformSource64Bytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitTransformSource64(sink);
      },
      bytes);
}

} // namespace rund::node::accel::detail
