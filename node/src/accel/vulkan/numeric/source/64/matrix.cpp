#include "../../matrix.hpp"
#include "../../source.hpp"

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitMatrixSource64(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return EmitNumericBaseSource(sink, true) &&
         AppendMatrixProgramSource(sink, R"GLSL(
#define RUND_MATRIX_SCALAR int64_t
#define RUND_MATRIX_INDEX(value) ix64(value)
#define RUND_MATRIX_MIDX midx64
#define RUND_MATRIX_ZERO int64_t(0)
#define RUND_MATRIX_ADD matrix_add_i64
#define RUND_MATRIX_MUL matrix_mul_i64
)GLSL");
}

} // namespace

[[nodiscard]] std::string MatrixSource64() {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [](auto &sink)
      noexcept(noexcept(EmitMatrixSource64(sink))) {
    return EmitMatrixSource64(sink);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool MatrixSource64Bytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitMatrixSource64(sink);
      },
      bytes);
}

} // namespace rund::node::accel::detail
