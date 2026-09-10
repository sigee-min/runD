#pragma once

#include "../../../device/residency/execution/graph_drain.hpp"
#include "../../../device/residency/execution/graph_persist.hpp"
#include "../../../device/residency/execution/graph_promote.hpp"
#include "../../../device/residency/pool.hpp"
#include "../../../pipeline/local.hpp"
#include "../../run/projection.hpp"
#include "../forecast.hpp"
#include "receipts.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail::graph_reduce {

struct HostSupply final {
  std::array<residency::PrefetchedPage,
             residency::execution::GraphPromoteCapacity>
      pages{};
  std::size_t page_count{};
  std::uint64_t fetched_pages{};
  std::uint64_t backing_bytes{};
  bool speculative{};
};

struct Interval final {
  std::uint64_t started{};
  std::uint64_t completed{};
};

// One running prefix may hide every external-input Forecast wait for the next
// ticket after its own Drain/Persist pair has already entered the timeline.
// This is a fixed topology bound, independent of the batch count Q.
inline constexpr std::size_t GraphTimelineConcurrentCapacity =
    residency::execution::GraphPromoteSourceCapacity + 2u;
static_assert(GraphTimelineConcurrentCapacity == 9u);

struct Timeline final {
  std::array<Interval, GraphTimelineConcurrentCapacity> concurrent{};
  std::array<Interval, 3u> transfers{};
  enum class Direction : std::uint8_t { HostToDevice, DeviceToHost };
  std::array<Direction, 3u> transfer_directions{};
  std::size_t concurrent_count{};
  std::size_t transfer_count{};
};

enum class TicketPhase : std::uint8_t {
  Empty,
  Projected,
  SupplyReady,
  PrefixReady,
  PrefixRunning,
  IntermediateDirty,
  CollectiveReady,
  CollectiveRunning,
  DeviceOutputDirty,
  OutputDraining,
  HostOutputDirty,
  OutputPersisting,
};

enum class ExecutionStage : std::uint8_t { None, Prefix, Collective };

struct LiveResource final {
  std::array<residency::CacheKey, PipelineLeafCapacity> keys{};
  residency::FrameRegion region{};
  std::size_t count{};
  bool dirty{};
};

struct StageEffects final {
  std::array<std::array<residency::CacheKey, PipelineLeafCapacity>,
             residency::TiledGraphResourceCapacity>
      keys{};
  std::array<residency::FrameRegion, residency::TiledGraphResourceCapacity>
      regions{};
  std::array<std::size_t, residency::TiledGraphResourceCapacity> counts{};
  std::array<bool, residency::TiledGraphResourceCapacity> retired{};
  std::array<bool, residency::TiledGraphResourceCapacity> written{};
};

struct StageScratch final {
  std::array<residency::PageUse,
             residency::TiledGraphPortCapacity * PipelineLeafCapacity>
      uses{};
  std::array<residency::GraphPortRequest, residency::TiledGraphPortCapacity>
      requests{};
  residency::Epoch epoch{};
  std::size_t use_count{};
  std::size_t port_count{};
  std::size_t anchor_port{};
};

struct Ticket final {
  residency::PageRun pages{};
  residency::Epoch prefix_epoch{};
  residency::Epoch collective_epoch{};
  VirtualEpochProjection byte_epoch{};
  std::shared_ptr<PipelineState> prefix;
  std::shared_ptr<PipelineState> collective;
  std::array<residency::PageUse, PipelineLeafCapacity> prefix_sources{};
  std::array<residency::PageUse, PipelineLeafCapacity> prefix_outputs{};
  std::array<residency::PageUse,
             residency::TiledGraphPortCapacity * PipelineLeafCapacity>
      prefix_uses{};
  std::array<residency::GraphPortRequest, residency::TiledGraphPortCapacity>
      prefix_requests{};
  std::array<residency::PageUse,
             residency::TiledGraphPortCapacity * PipelineLeafCapacity>
      collective_uses{};
  std::array<residency::GraphPortRequest, residency::TiledGraphPortCapacity>
      collective_requests{};
  std::array<residency::PageUse, PipelineLeafCapacity> collective_outputs{};
  std::array<residency::CacheKey, PipelineLeafCapacity> output_keys{};
  std::array<residency::CacheBinding,
             residency::TiledGraphPortCapacity * PipelineLeafCapacity>
      prefix_bindings{};
  std::array<residency::CacheTransition,
             residency::TiledGraphPortCapacity * PipelineLeafCapacity * 3u>
      prefix_transitions{};
  std::array<residency::GraphLeasePort, residency::TiledGraphPortCapacity>
      prefix_ports{};
  std::array<residency::GraphPageRemap, residency::GraphPageRemapCapacity>
      prefix_remaps{};
  std::array<residency::CacheBinding,
             residency::TiledGraphPortCapacity * PipelineLeafCapacity>
      collective_bindings{};
  std::array<residency::CacheTransition,
             residency::TiledGraphPortCapacity * PipelineLeafCapacity * 3u>
      collective_transitions{};
  std::array<residency::GraphLeasePort, residency::TiledGraphPortCapacity>
      collective_ports{};
  std::array<residency::GraphPageRemap, residency::GraphPageRemapCapacity>
      collective_remaps{};
  std::array<residency::CacheBinding, PipelineLeafCapacity> resident_outputs{};
  std::array<LiveResource, residency::TiledGraphResourceCapacity>
      live_resources{};
  HostSupply host_supply{};
  residency::FrameRegion input_region{};
  residency::FrameRegion intermediate_region{};
  residency::FrameRegion output_region{};
  residency::FrameRegion resident_output_region{};
  Timeline prefix_timeline{};
  Timeline collective_timeline{};
  std::size_t count{};
  std::size_t prefix_binding_count{};
  std::size_t prefix_transition_count{};
  std::size_t prefix_port_count{};
  std::size_t prefix_remap_count{};
  std::size_t prefix_use_count{};
  std::size_t prefix_request_count{};
  std::size_t prefix_anchor_port{};
  std::size_t collective_binding_count{};
  std::size_t collective_transition_count{};
  std::size_t collective_port_count{};
  std::size_t collective_remap_count{};
  std::size_t collective_use_count{};
  std::size_t collective_request_count{};
  std::size_t collective_anchor_port{};
  std::size_t collective_output_port{};
  std::uint64_t batch{};
  // Transient CPU receipt identity; Authority's
  // CycleAuthorityState::graph_persists remains the sole retry journal and
  // this field is never a retry mirror.
  std::uint64_t book_domain{};
  std::array<residency::execution::GraphReady,
             residency::execution::GraphPromoteSourceCapacity>
      host_ready{};
  residency::execution::GraphPromote input_promote{};
  residency::execution::GraphDrain output_drain{};
  residency::execution::GraphPersist output_persist{};
  std::uint32_t forecast_stage{};
  std::array<std::uint32_t, residency::execution::GraphPromoteSourceCapacity>
      forecast_resources{};
  std::size_t host_ready_count{};
  std::uint64_t prefix_token{};
  std::uint64_t collective_token{};
  CpuEpochReceipt prefix_receipt{};
  CpuEpochReceipt collective_receipt{};
  CpuEpochReceipt supply_receipt{};
  CpuEpochReceipt middle_receipt{};
  std::uint32_t bank{};
  TicketPhase phase{TicketPhase::Empty};
  ExecutionStage submitted{ExecutionStage::None};
  bool prefix_executed{};
  bool collective_executed{};
  bool prefix_stats_folded{};
  bool collective_stats_folded{};
  bool intermediate_dirty{};
  bool output_dirty{};
  bool poison{};
};

// Reconstructs one quiescent fixed ticket without requiring move assignment
// from any callback-return-gated physical capability.
[[nodiscard]] bool reset_ticket(Ticket &) noexcept;

} // namespace rund::compute::detail::graph_reduce
