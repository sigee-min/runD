#include "../source.hpp"

#include "solve/direct.hpp"
#include "solve/factor.hpp"

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitSolveSource(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return EmitNumericBaseSource(sink, false) &&
         sink.append(source::solve::Factor) &&
         sink.append(source::solve::Direct);
}

} // namespace

[[nodiscard]] std::string SolveSource() {
  std::uint64_t exact_bytes = 0u;
  const auto emit = [](auto &sink)
      noexcept(noexcept(EmitSolveSource(sink))) {
    return EmitSolveSource(sink);
  };
  return backend_source_recipe::bytes(emit, exact_bytes)
             ? backend_source_recipe::materialize(emit, exact_bytes)
             : std::string{};
}

bool SolveSourceBytes(std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [](backend_source_recipe::CountSink &sink) noexcept {
        return EmitSolveSource(sink);
      },
      bytes);
}

} // namespace rund::node::accel::detail
