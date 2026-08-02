#pragma once

#include <rund/compute/pipeline/coordinate.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::compute {

struct MemoryBudget final {
  // Maximum complete runD-owned preparation reservation. The check covers
  // Pipeline state/workspace, dense backend View and primitive scratch
  // Buffers, retained host/CPU execution storage, and explicitly sized native
  // command/descriptor storage before those owners are materialized. Opaque
  // allocator headers and driver-private allocation granularity remain
  // post-prepare telemetry through memory().
  std::uint64_t bytes{};
};

struct PipelinePlan final {
  // Existing caller-owned logical Buffer payload referenced by this Pipeline.
  std::uint64_t persistent_bytes{};
  std::uint64_t state_bytes{};
  std::uint64_t transient_bytes{};
  // Complete runD-owned cold preparation reservation.  It is the checked sum
  // of the disjoint Buffer, host, CPU tile-run, and native product-owned
  // components below and participates in MemoryBudget before materialization.
  std::uint64_t prepared_bytes{};
  // Dense View normalization and primitive scratch Buffer payload.
  std::uint64_t prepared_buffer_bytes{};
  // Retained Pipeline/Job route metadata and CPU Program-private run state.
  std::uint64_t prepared_host_bytes{};
  // CPU worker, tile-executor, collective, and primitive scratch payload.
  std::uint64_t prepared_tile_bytes{};
  // Backend command/descriptor storage whose byte ownership is visible to
  // runD. Metal Pipeline ICB bytes are exact device-calibrated size-class
  // allocations and are reserved here before materialization. Other opaque
  // driver bookkeeping that has no allocation query remains post-prepare
  // telemetry.
  std::uint64_t prepared_native_bytes{};
  // Backend primitive scratch is one serially reused arena. scratch_bytes is
  // its retained payload and scratch_count is its physical page count.
  std::uint64_t scratch_bytes{};
  std::uint64_t scratch_count{};
  // Bytes copied by final and append-only window publication. This is traffic,
  // not retained storage, and therefore does not participate in peak_bytes.
  std::uint64_t publish_bytes{};
  // Complete checked retained admission reservation. total_bytes is the
  // checked sum of referenced caller payload and this Pipeline-owned
  // reservation. Host/native structural envelopes may materialize below their
  // reserved byte counts, never above them.
  std::uint64_t peak_bytes{};
  // Exact byte extent of explicit Pipeline-owned sealed arena mappings before
  // host-page rounding. This excludes caller-owned persistent Buffers and
  // never includes process RSS, allocator metadata, or driver-private state.
  std::uint64_t arena_extent_bytes{};
  // Conservative backing charge admitted by the Device Pipeline budget. It is
  // the checked sum of each explicit mapping rounded once to its platform page
  // boundary plus separately owned exact commitments; it is never derived by
  // rounding peak_bytes as one aggregate allocation.
  std::uint64_t committed_peak_bytes{};
  std::uint64_t total_bytes{};
  // These three reports exclude persistent_bytes and include the same fixed
  // infrastructure B = state_bytes + prepared_bytes. logical_bytes adds every
  // expanded Program logical workspace occurrence, live_bytes adds the
  // largest executable Program live workspace, and physical_bytes adds the
  // retained transient workspace. physical_bytes is exactly peak_bytes; none
  // of these reports is clamped to another report.
  std::uint64_t logical_bytes{};
  std::uint64_t live_bytes{};
  std::uint64_t physical_bytes{};
  std::uint64_t allocation_count{};
  std::uint64_t reuse_count{};
  std::uint64_t publish_count{};
  // Compact nested-schedule shape and admitted implementation counts.
  std::uint64_t outer_window_count{};
  std::uint64_t tile_capacity{};
  std::uint64_t inner_iteration_count{};
  std::uint64_t node_count{};
  std::uint64_t resource_count{};
  // Exact nonzero boundaries in the compact frozen schedule. This is derived
  // from resource hazards and shared-workspace reuse, never command count.
  std::uint64_t barrier_count{};
  std::uint64_t prepared_template_count{};
  std::uint64_t prepared_command_count{};
  // Largest single Program workspace before cross-step reuse. These fields
  // identify an oversized owner without exposing backend addresses.
  std::uint64_t largest_bytes{};
  std::size_t largest_step{std::numeric_limits<std::size_t>::max()};
  std::size_t largest_iteration{std::numeric_limits<std::size_t>::max()};
  std::size_t largest_outer_window{std::numeric_limits<std::size_t>::max()};
  std::size_t largest_inner_iteration{std::numeric_limits<std::size_t>::max()};
  PipelineNestedPhase largest_nested_phase{PipelineNestedPhase::None};
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
  std::size_t view_outer_window{std::numeric_limits<std::size_t>::max()};
  std::size_t view_inner_iteration{std::numeric_limits<std::size_t>::max()};
  PipelineNestedPhase view_nested_phase{PipelineNestedPhase::None};
  std::size_t view_binding{std::numeric_limits<std::size_t>::max()};
  std::size_t peak_step{std::numeric_limits<std::size_t>::max()};
  std::size_t peak_iteration{std::numeric_limits<std::size_t>::max()};
  std::size_t peak_outer_window{std::numeric_limits<std::size_t>::max()};
  std::size_t peak_inner_iteration{std::numeric_limits<std::size_t>::max()};
  PipelineNestedPhase peak_nested_phase{PipelineNestedPhase::None};

  [[nodiscard]] constexpr bool
  operator==(const PipelinePlan &) const noexcept = default;
};

static_assert(std::is_trivially_copyable_v<MemoryBudget>);
static_assert(std::is_trivially_copyable_v<PipelinePlan>);

} // namespace rund::compute
