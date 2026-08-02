#include "../../source.hpp"

#include "solve/direct.hpp"
#include "solve/factor.hpp"

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitSolveSource64(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return EmitNumericBaseSource(sink, true) &&
         sink.append(source::lane64::solve::Factor) &&
         sink.append(source::lane64::solve::Direct);
}

} // namespace

[[nodiscard]] std::string SolveSource64() {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [](auto &sink)
      noexcept(noexcept(EmitSolveSource64(sink))) {
    return EmitSolveSource64(sink);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool SolveSource64Bytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitSolveSource64(sink);
      },
      bytes);
}

} // namespace rund::node::accel::detail
