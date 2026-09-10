#pragma once

#include "../pipeline/residency/model.hpp"
#include "../pipeline/state.hpp"
#include "graph/reduce/failure.hpp"
#include "graph/reduce/receipts.hpp"
#include "host_ring.hpp"

#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rund::compute::detail {

namespace sliding_product_detail {
struct SlidingProductOwner;
}

namespace device_vsm_product_detail {
struct DeviceVsmTestBarrier;
}

struct VirtualBufferState final {
  std::shared_ptr<VirtualBacking> backing;
  Type type{Type::I32};
  FixedFormat format{};
  std::uint64_t count{};
  std::uint64_t element_bytes{};
  std::uint64_t bytes{};
};

enum class VirtualRoute : std::uint8_t {
  Pointwise,
  MultiPointwise,
  Window,
  Reduction,
  Scan,
  GraphPointwise,
  GraphReduction,
};

struct VirtualGeometry final {
  VirtualRoute route{VirtualRoute::Pointwise};
  std::uint64_t input_payload_elements{};
  std::uint64_t output_payload_elements{};
  std::uint64_t input_frame_elements{};
  std::uint64_t output_frame_elements{};
  std::uint64_t intermediate_frame_elements{};
  std::uint64_t input_prefix_elements{};
  std::uint64_t output_prefix_elements{};
  std::uint32_t operation{};
  std::uint32_t boundary{};
  bool device_vsm_required{};
  std::uint64_t materialization_hi{};
  std::uint64_t materialization_lo{};
};

// The preparation boundary is the only owner of the bounded Window decision.
// Route code consumes this immutable value; it must not repeat the geometry or
// endpoint predicates after the pool and pipelines have been created.
enum class VirtualWindowPreflightMode : std::uint8_t {
  Declined,
  WindowRing,
  RollingPool,
};

enum class VirtualWindowPreflightEndpoint : std::uint8_t {
  Invalid,
  Resident,
  Staged,
};

struct VirtualWindowPreflight final {
  VirtualWindowPreflightMode mode{VirtualWindowPreflightMode::Declined};
  VirtualWindowPreflightEndpoint endpoint{
      VirtualWindowPreflightEndpoint::Invalid};
  std::uint64_t page_count{};
  std::uint64_t frame_capacity{};
  std::uint64_t input_bytes{};
  std::uint64_t output_bytes{};
  std::uint64_t input_frame_bytes{};
  std::uint64_t output_frame_bytes{};
  std::uint64_t frame_bytes{};
  std::uint64_t device_storage_bytes{};
  VirtualHostRingCapacities host{};

  [[nodiscard]] constexpr bool
  operator==(const VirtualWindowPreflight &) const noexcept = default;
};

enum class VirtualPipelinePhase : std::uint8_t {
  Ready,
  Running,
  Poisoned,
};

struct VirtualPipelineState final {
  // Canonical public-input authority. Input rows are stored in Program-port
  // order and are never mirrored by a scalar alias. Seven leaves plus the
  // single public output fit the fixed DeviceVsm resident capacity of eight;
  // wider signatures fail before any physical owner is allocated.
  static constexpr std::size_t InputCapacity = 7u;
  std::array<std::shared_ptr<VirtualBufferState>, InputCapacity> inputs{};
  std::size_t input_count{};
  std::shared_ptr<VirtualBufferState> output;
  std::shared_ptr<PipelineState> pipeline;
  std::shared_ptr<PipelineState> alternate_pipeline;
  // Canonical stage-major, bank-minor prepared execution table. Graph stage
  // and terminal selection always reads this table through the narrow
  // accessors below; ordinary routes continue to use the two bank aliases.
  std::vector<std::shared_ptr<PipelineState>> graph_pipelines;
  // External Graph resources in canonical Program-input order. Planner
  // resource order is a physical-placement fact and must not be reused as a
  // semantic binding order when multiple public backings are present.
  std::array<std::uint32_t, InputCapacity> graph_input_resources{};
  std::size_t graph_input_resource_count{};
  // Optional accelerator-only Pipeline containing the complete typed Map DAG
  // used by the one-submit DeviceVsm route. It is never substituted for the
  // physical Host-wavefront stage table above.
  std::shared_ptr<PipelineState> device_vsm_semantic_pipeline;
  // Accelerator-only cold owner for the aggregate DeviceVsm product. The CPU
  // run path never reads this field. Exact per-run Authority credentials and
  // Pipeline snapshots remain outside the cache and are minted on every run.
  std::shared_ptr<void> device_vsm_product_cache;
  // Persistent Sliding owns one sealed shape cache. Per-run Authority
  // credentials and callback/request fields remain in the transient run.
  std::shared_ptr<sliding_product_detail::SlidingProductOwner>
      sliding_product_cache;
  // CPU Graph credentials survive ticket reconstruction in this fixed,
  // allocation-free run book. The ticket fields are role/bank handles only.
  std::shared_ptr<graph_reduce::CpuReceiptBook> cpu_receipts;
  // Allocated during graph preparation, before CPU Authority mutation. Abort
  // only populates this holder; it never allocates on a failure path.
  std::shared_ptr<graph_reduce::CpuGraphQuarantine> cpu_quarantine_hold;
  // One fixed first-failure diagnostic for the current graph attempt. It owns
  // no rows and never changes execution or cleanup policy.
  graph_reduce::FailLog failure_log{};
  VirtualGeometry geometry{};
  VirtualWindowPreflight window_preflight{};
  Stats stats{};
  mutable std::mutex gate;
  // Source-private evidence hook. A null barrier is the product default and
  // leaves the native callback path unchanged.
  device_vsm_product_detail::DeviceVsmTestBarrier *device_vsm_test_barrier{};
  VirtualPipelinePhase phase{VirtualPipelinePhase::Ready};
  enum class SampleState : std::uint8_t { Inactive, Active };
  SampleState samples{SampleState::Inactive};

  ~VirtualPipelineState() noexcept;
};

[[nodiscard]] std::shared_ptr<PipelineState>
graph_stage_pipeline(const VirtualPipelineState &, std::size_t stage,
                     std::size_t bank) noexcept;

[[nodiscard]] std::shared_ptr<PipelineState>
graph_terminal_pipeline(const VirtualPipelineState &, std::size_t bank) noexcept;

[[nodiscard]] bool cpu_graph_ready(VirtualPipelineState &) noexcept;

[[nodiscard]] inline VirtualBufferState *
virtual_input(VirtualPipelineState &state,
              const std::size_t index = 0u) noexcept {
  return index < state.input_count ? state.inputs[index].get() : nullptr;
}

[[nodiscard]] inline const VirtualBufferState *
virtual_input(const VirtualPipelineState &state,
              const std::size_t index = 0u) noexcept {
  return index < state.input_count ? state.inputs[index].get() : nullptr;
}

} // namespace rund::compute::detail
