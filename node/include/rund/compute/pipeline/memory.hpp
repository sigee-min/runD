#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::compute {

struct MemoryBudget final {
  // Maximum planned bytes retained by Pipeline-owned state, Program
  // workspace, dense backend View storage, and primitive scratch arena.
  // Native driver metadata and physical allocation granularity remain
  // observable through memory().
  std::uint64_t bytes{};
};

struct PipelinePlan final {
  // Existing caller-owned logical Buffer payload referenced by this Pipeline.
  std::uint64_t persistent_bytes{};
  std::uint64_t state_bytes{};
  std::uint64_t transient_bytes{};
  // Backend storage referenced by prepared native commands. Dense View
  // normalization and primitive scratch are allocated by this immutable
  // Pipeline plan, never by private Pipeline Jobs.
  std::uint64_t prepared_bytes{};
  // Backend primitive scratch is one serially reused arena. scratch_bytes is
  // its retained payload and scratch_count is its physical page count.
  std::uint64_t scratch_bytes{};
  std::uint64_t scratch_count{};
  // Bytes copied by terminal publication. This is traffic, not retained
  // storage, and therefore does not participate in peak_bytes.
  std::uint64_t publish_bytes{};
  // Exact planned retained payload. total_bytes is exactly referenced caller
  // payload plus this Pipeline-owned payload.
  std::uint64_t peak_bytes{};
  std::uint64_t total_bytes{};
  std::uint64_t allocation_count{};
  std::uint64_t reuse_count{};
  std::uint64_t publish_count{};
  // Largest single Program workspace before cross-step reuse. These fields
  // identify an oversized owner without exposing backend addresses.
  std::uint64_t largest_bytes{};
  std::size_t largest_step{std::numeric_limits<std::size_t>::max()};
  std::size_t largest_iteration{std::numeric_limits<std::size_t>::max()};
  std::size_t largest_chunk{std::numeric_limits<std::size_t>::max()};
  // Largest backend-normalized View and its public Pipeline coordinates.
  // binding is the canonical Program graph-binding ordinal.
  std::uint64_t view_bytes{};
  std::uint64_t view_span_bytes{};
  std::uint64_t view_backing_bytes{};
  std::uint64_t view_offset_bytes{};
  std::uint64_t view_stride_bytes{};
  std::uint64_t view_element_bytes{};
  std::uint64_t view_count{};
  std::uint64_t view_alignment{};
  std::size_t view_step{std::numeric_limits<std::size_t>::max()};
  std::size_t view_iteration{std::numeric_limits<std::size_t>::max()};
  std::size_t view_binding{std::numeric_limits<std::size_t>::max()};
  std::size_t peak_step{std::numeric_limits<std::size_t>::max()};
  std::size_t peak_iteration{std::numeric_limits<std::size_t>::max()};

  [[nodiscard]] constexpr bool
  operator==(const PipelinePlan &) const noexcept = default;
};

static_assert(std::is_trivially_copyable_v<MemoryBudget>);
static_assert(std::is_trivially_copyable_v<PipelinePlan>);

} // namespace rund::compute
