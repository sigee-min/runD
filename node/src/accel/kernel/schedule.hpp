#pragma once

#include "../context/internal.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

struct ScheduledStepOrder {
  std::vector<std::uint8_t> barriers{};
  std::size_t count = 0u;
  bool ok = true;

  [[nodiscard]] std::size_t size() const noexcept { return count; }
  [[nodiscard]] std::size_t at(const std::size_t index) const noexcept {
    return index;
  }
  [[nodiscard]] bool barrier_before(const std::size_t index) const noexcept {
    return index < barriers.size() && barriers[index] != 0u;
  }
};

[[nodiscard]] ScheduledStepOrder BuildScheduledStepOrder(
    std::span<const KernelExecutionStep> steps,
    std::span<const rund::kernel::BufferRole> graph_roles,
    std::span<const std::uint64_t> graph_alias_representatives,
    std::span<const std::uint8_t> required_barriers);

} // namespace rund::node::accel::detail
