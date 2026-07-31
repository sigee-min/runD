#pragma once

#include "backend/run.hpp"

#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

enum class MapRecurrenceState : std::uint8_t {
  Ineligible,
  Ready,
  Invalid,
};

// Cold evidence for replacing a complete element-local Map recurrence with
// one backend kernel. Ineligible is not an error: the canonical prepared
// command stream remains authoritative. Invalid means the chain matched the
// recurrence semantics but its retained source could not be transformed
// exactly, so preparation must fail closed.
struct MapRecurrence final {
  MapRecurrenceState state = MapRecurrenceState::Ineligible;
  const BoundStep *first = nullptr;
  const BoundStep *last = nullptr;
  rund::kernel::BindingSet bindings{};
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::ComputePlan plan{};
  const rund::kernel::ComputeDispatchWindow *windows = nullptr;
  std::uint64_t window_count = 0u;
  std::uint64_t iterations = 0u;
  const char *reason = "compute_pipeline_recurrence_ineligible";

  [[nodiscard]] constexpr bool ready() const noexcept {
    return state == MapRecurrenceState::Ready;
  }

  [[nodiscard]] constexpr bool invalid() const noexcept {
    return state == MapRecurrenceState::Invalid;
  }
};

// Eligibility depends only on graph semantics and resident identity. It never
// branches on tile count, iteration count, device family, or measured timing.
[[nodiscard]] MapRecurrence
BuildMapRecurrence(std::span<const BackendBatchEntry> entries,
                   std::span<const std::uint8_t> barriers);

} // namespace rund::node::accel::detail
