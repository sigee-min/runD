#include "../matrix.hpp"
#include "../source.hpp"

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitMatrixSource(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return EmitNumericBaseSource(sink, false) &&
         AppendMatrixProgramSource(sink, R"GLSL(
#define RUND_MATRIX_SCALAR int
#define RUND_MATRIX_INDEX(value) uint(value)
#define RUND_MATRIX_MIDX midx
#define RUND_MATRIX_ZERO 0
#define RUND_MATRIX_ADD matrix_add_i32
#define RUND_MATRIX_MUL matrix_mul_i32
)GLSL");
}

} // namespace

[[nodiscard]] std::string MatrixSource() {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [](auto &sink)
      noexcept(noexcept(EmitMatrixSource(sink))) {
    return EmitMatrixSource(sink);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool MatrixSourceBytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitMatrixSource(sink);
      },
      bytes);
}

} // namespace rund::node::accel::detail
