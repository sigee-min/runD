#pragma once

#include <rund/compute/graph/info.hpp>
#include <rund/compute/status.hpp>

#include <cstdint>
#include <span>

namespace rund::compute::detail::resource_detail {

struct Domain final {
  std::uint32_t count{};
  std::uint32_t predicate{};
  std::uint64_t expected{};

  [[nodiscard]] constexpr bool empty() const noexcept {
    return count == 0u && predicate == 0u;
  }
  [[nodiscard]] constexpr bool
  operator==(const Domain &) const noexcept = default;
};

enum class Write : std::uint8_t {
  Partial,
  Full,
  Domain,
};

enum class Inplace : std::uint8_t {
  None,
  Pointwise,
};

struct MemoryNode final {
  Domain domain{};
  Write write{Write::Partial};
  Inplace inplace{Inplace::None};
};

[[nodiscard]] Status plan_memory(graph::Info &info,
                                 std::span<const MemoryNode> nodes,
                                 std::uint64_t page_bytes) noexcept;

} // namespace rund::compute::detail::resource_detail
