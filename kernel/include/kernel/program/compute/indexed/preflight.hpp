#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::kernel {

// A single immutable control pipeline has two admitted dispatch shapes:
// one group validates/publishes directly, or G groups publish private minima
// followed by one group reducing them. No group waits for another group.
struct IndexPreflightPlan final {
  u32 group_count{};
  u32 pass_count{};
  u64 partial_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const IndexPreflightPlan &) const noexcept = default;
};

[[nodiscard]] constexpr IndexPreflightPlan
PlanIndexPreflight(const u64 capacity) noexcept {
  if (capacity == 0u || capacity > ~u32{0u}) {
    return {};
  }
  if (capacity <= 65536u) {
    return {1u, 1u, 0u};
  }
  const u64 requested = capacity / 4096u + (capacity % 4096u != 0u ? 1u : 0u);
  const u32 groups = static_cast<u32>(requested < 1024u ? requested : 1024u);
  return {groups, 2u, static_cast<u64>(groups) * sizeof(u32)};
}

} // namespace rund::kernel
