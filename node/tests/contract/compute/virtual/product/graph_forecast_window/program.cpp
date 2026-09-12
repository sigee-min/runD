#include "internal.hpp"

namespace rund_node_test_virtual::product::graph_forecast_window {
namespace {
template <std::size_t First, std::size_t Count, class Expression>
[[nodiscard]] constexpr auto expand(Expression value) {
  if constexpr (Count == 1u) {
    return value + std::uint64_t{First};
  } else {
    constexpr std::size_t Half = Count / 2u;
    return expand<First, Half>(value) +
           expand<First + Half, Count - Half>(value);
  }
}
} // namespace
rund::compute::Result<Program>
build_program(const rund::compute::Device &device) {
  return rund::compute::on(device)
      .input<std::uint64_t>(PageElements)
      .zip_input<std::uint64_t>(PageElements)
      .zip_input<std::uint64_t>(PageElements)
      .zip_input<std::uint64_t>(PageElements)
      .branch([](auto a, auto b, auto c, auto d) {
        const auto prefix = a.map("forecast-prefix",
                                  [](auto v) { return expand<1u, Leaves>(v); });
        const auto slow = b.map("forecast-slow", [](auto v) {
          return expand<Leaves + 1u, Leaves>(v);
        });
        const auto fast = c.map("forecast-fast", [](auto v) {
          return expand<2u * Leaves + 1u, Leaves>(v);
        });
        const auto refill = d.map("forecast-refill", [](auto v) {
          return expand<3u * Leaves + 1u, Leaves>(v);
        });
        return zip(prefix, slow, fast, refill)
            .map("forecast-final", [](auto av, auto bv, auto cv, auto dv) {
              return av + bv + cv + dv;
            });
      })
      .compile();
}
std::uint64_t expected(const std::size_t element) noexcept {
  std::uint64_t result = 0u;
  for (std::size_t input = 0u; input < Inputs; ++input) {
    result += (element + input + 1u) * Leaves + Leaves * (Leaves + 1u) / 2u +
              input * Leaves * Leaves;
  }
  return result;
}
} // namespace rund_node_test_virtual::product::graph_forecast_window
