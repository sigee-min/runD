#pragma once

#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <span>

namespace rund_node_flow_contract {

template <class T, std::size_t N> struct ContiguousInput final {
  T values[N]{};

  [[nodiscard]] constexpr T *begin() noexcept { return values; }
  [[nodiscard]] constexpr const T *begin() const noexcept { return values; }
  [[nodiscard]] constexpr T *end() noexcept { return values + N; }
  [[nodiscard]] constexpr const T *end() const noexcept { return values + N; }
  [[nodiscard]] constexpr T *data() noexcept { return values; }
  [[nodiscard]] constexpr const T *data() const noexcept { return values; }
  [[nodiscard]] static constexpr std::size_t size() noexcept { return N; }
};

struct FlowHash final {
  std::uint64_t graph{};
  std::uint64_t output{};
};

[[nodiscard]] int CheckExpressions(rund::compute::Backend, std::uint64_t &,
                                   std::uint64_t &);
[[nodiscard]] int CheckRecords(rund::compute::Backend);
[[nodiscard]] bool CheckComposition(rund::compute::Backend,
                                    std::array<FlowHash, 6u> &);
[[nodiscard]] bool CheckTyped(rund::compute::Backend,
                              std::array<FlowHash, 3u> &);
[[nodiscard]] int
    CheckBackendContracts(std::span<const rund::compute::Backend>);
[[nodiscard]] int CheckParityBackends(std::span<const rund::compute::Backend>);
[[nodiscard]] int CheckBasic();
[[nodiscard]] int CheckShape();
[[nodiscard]] int CheckDevice();

} // namespace rund_node_flow_contract
