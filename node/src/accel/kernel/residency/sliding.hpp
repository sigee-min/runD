#pragma once

#include "../callback.hpp"
#include "window.hpp"

#include <accel/check.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

// A sliding lowering has fixed native storage. Logical coordinates may be
// arbitrarily large; only these reusable slots cross the backend boundary.
inline constexpr std::size_t ResidencySlidingCapacity = 4u;

enum class ResidencySlidingMemory : std::uint8_t {
  HostCoherent,
  NoncoherentIntegratedCopy,
  NoncoherentSplitTransfer,
};

// Backend facts, not cache or page policy. `retained_bytes` and
// `transient_bytes` describe the incremental lowering payload; already
// admitted prepared-Pipeline owners are deliberately excluded.
struct BackendResidencySlidingCapability final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  ResidencySlidingMemory memory{ResidencySlidingMemory::HostCoherent};
  std::uint64_t retained_bytes{};
  std::uint64_t transient_bytes{};
  std::uint8_t max_slots{};
  bool callbacks_async{};
  bool descriptor_release_acquire{};
  bool flush_before_frontier{};
  bool integrated_copy{};
  bool distinct_transfer_queue{};
};

// Immutable Authority-selected native work. The backend may authenticate and
// execute these values, but may not derive pages, victims, next-use, Graph
// edges, or a replacement policy from them.
struct BackendResidencySlidingBatch final {
  std::shared_ptr<void> prepared{};
  std::array<std::uint32_t, ResidencyWindowLocalCapacity> locals{};
  std::size_t local_count{};
  std::uint64_t coordinate{};
  std::uint64_t turn{};
  std::uint64_t read_mask{};
  std::uint64_t write_mask{};
  std::uint64_t descriptor_generation{};
  std::uint32_t control_generation{};
  std::uint8_t slot{};
};

// Descriptor publication is a concrete backend operation. A capability may
// claim release/acquire only when this exact row is release-published before
// submission and acquire-validated by the native submit hook.
struct BackendResidencySlidingDescriptor final {
  const void *owner{};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t coordinate{};
  std::uint64_t turn{};
  std::uint64_t read_mask{};
  std::uint64_t write_mask{};
  std::uint64_t descriptor_generation{};
  std::uint32_t control_generation{};
  std::uint8_t stride{};
  std::uint8_t slot{};
};

struct BackendResidencySlidingTerminal final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  NativeTerminal terminal{NativeTerminal::Known};
  std::uint64_t coordinate{};
  std::uint64_t turn{};
  std::uint64_t queue_calls{};
  std::uint8_t slot{};
  bool dispatched{};
  bool completed{};
  bool may_write{};
};

struct BackendResidencySlidingFinal final {
  rund::AccelCheck check{false, "accel_kernel_pipeline_invalid"};
  NativeTerminal terminal{NativeTerminal::Known};
  std::uint64_t plan_identity{};
  std::uint64_t token{};
  std::uint64_t generation{};
  std::uint64_t coordinate_count{};
  std::uint64_t accepted_coordinates{};
  std::uint64_t released_coordinates{};
  std::uint64_t queue_calls{};
  std::uint64_t native_inflight_peak{};
  std::uint64_t retained_bytes{};
  std::uint64_t transient_bytes{};
  std::uint64_t first_failure_coordinate{};
  std::uint64_t completed_ns{};
};

} // namespace rund::node::accel::detail
