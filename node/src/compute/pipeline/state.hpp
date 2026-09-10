#pragma once

#include "../../accel/kernel/prepared/pipeline.hpp"
#include "../../accel/kernel/prepared/template/registry.hpp"

#include "state/base.hpp"
#include "state/memory.hpp"
#include "state/plan.hpp"
#include "state/publication.hpp"
#include "state/route.hpp"
#include "state/window.hpp"

#include "../device/state.hpp"
#include "../job/state.hpp"
#include "../program/state.hpp"
#include "service_free_direct/model.hpp"

#include <rund/compute/pipeline/profile.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace rund::compute::detail {

// Protected by PipelineState::gate. Replaced as a unit when claims succeed;
// native/observation identities and quarantine deliberately outlive this scope.
struct PipelineAttemptState final {
  std::uint64_t generation{};
  std::size_t verified{};
  std::uint8_t parity{};
  bool failure_step_known{};
  bool writes_possible{};
  bool backend_submitted{};
  bool dispatch_timing{};
};

struct PipelineProfileState final {
  std::vector<PipelineStepProfile> steps;
  std::vector<std::uint64_t> started_ns;
  std::vector<bool> started;
  std::uint64_t instrumentation_command_count{};
  std::uint64_t instrumentation_byte_count{};
};

struct PipelineResource final {
  static constexpr std::uint32_t no_output =
      std::numeric_limits<std::uint32_t>::max();

  std::shared_ptr<BufferState> buffer;
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t count{};
  std::size_t bytes{};
  std::size_t offset{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
  std::uint32_t output{no_output};
  std::uint32_t first_write{resource::NoNode};
  // Canonical transactional owner projection. `no_output` means self; paired
  // resources point at each other and every runtime consumer uses this map.
  std::uint32_t partner{no_output};
  bool owned{};
  bool terminal_publish{};
};

struct PipelineOutputState final {
  std::uint64_t generation{};
  std::uint64_t hash{};
  std::uint32_t resource{};
  bool observed{};
};

struct PipelineStep final {
  std::shared_ptr<ProgramState> program;
  std::shared_ptr<JobState> job;
  std::shared_ptr<JobState> alternate_job;
  std::uint32_t logical_step{};
  std::uint32_t iteration{};
  std::uint32_t iteration_bound{1u};
  // Binding descriptors and resource ordinals are cold planning authority and
  // are not retained after the two private Jobs/native streams are frozen.
  // Warm execution needs only whether this step can make a write observable.
  bool writes : 1 {};
  bool writes_each_iteration : 1 {};
  PipelineRoute route{PipelineRoute::Ordinary};
  // One-based index into PipelineState::windows.
  std::uint16_t window{};
};

static_assert(sizeof(PipelineStep) <=
              sizeof(std::shared_ptr<ProgramState>) * 4u);

struct PipelineDependency final {
  std::uint32_t before{};
  std::uint32_t after{};
  std::uint32_t resource{};
  PipelineAccess before_access{PipelineAccess::Read};
  PipelineAccess after_access{PipelineAccess::Read};
};

struct PipelineState final {
  std::shared_ptr<DeviceState> device;
  // Declared before all direct/native Buffer owners so reverse destruction
  // drops those aliases before the Pool releases Authority regions and refunds
  // physical capacity.
  std::shared_ptr<const residency::ResidencyPlan> residency;
  std::shared_ptr<residency::Pool> residency_pool;
  // Declared before all Pipeline-private owners so reverse destruction keeps
  // the aggregate Device charge live until those owners are gone.
  storage::Reservation private_memory;
  // Virtual-only backend submission resources are materialized after the
  // common physical plan, but before VirtualPipeline publication. This exact
  // Device admission remains live until every backend owner below is gone.
  storage::Reservation residency_submission_memory;
  std::shared_ptr<PipelinePublicationState> publication;
  // Canonical owner of the one CPU prepared mapping. Jobs and Program storage
  // retain lifetime references to this same control block, but planning,
  // observation, and destruction read this Pipeline-owned authority.
  std::shared_ptr<CpuPreparedArena> cpu_prepared_arena;
  std::vector<PipelineStep> steps;
  PipelineWindows windows;
  // Sealed rank of physical window steps in each prefix. Empty for a Pipeline
  // without resident windows; otherwise size is steps.size() + 1.
  std::vector<std::uint16_t> window_rank;
  std::vector<PipelineResource> resources;
  std::vector<std::shared_ptr<BufferState>> prepared_buffers;
  std::vector<std::shared_ptr<CpuGraphStorage>> cpu_storage;
  std::vector<BufferClaim> claims;
  std::vector<BufferClaim> alternate_claims;
  // Runtime retains the exact ordered cold plan. Admission clears only each
  // target's cold external_owner after PipelineResource takes ownership.
  std::vector<PipelinePublicationPlan> publications;
  std::vector<PipelineOutputState> outputs;
  // Lookup-only permutation of outputs, sorted by Buffer owner address. The
  // canonical output vector remains in resource order and owns hashing order.
  std::vector<std::uint32_t> output_lookup;
  std::vector<PipelineDependency> dependencies;
  std::vector<std::uint8_t> barriers;
  std::unique_ptr<PipelineProfileState> profile;
  // Sole mutable authority for Program-template publication across primary
  // and transactional-alternate native streams. Its limit is copied from the
  // frozen memory plan before private Jobs are materialized.
  node::accel::detail::PreparedKernelTemplateRegistry accel_templates;
  node::accel::detail::PreparedKernelPipeline prepared;
  node::accel::detail::PreparedKernelPipeline alternate_prepared;
  // Views into the Device-global Pool's resident frame arenas.
  std::uint32_t residency_input{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t residency_control{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t residency_output{std::numeric_limits<std::uint32_t>::max()};
  std::uint64_t residency_input_page_bytes{};
  std::uint64_t residency_output_page_bytes{};
  std::uint32_t residency_bank{};
  std::array<PipelineResidencyPort, residency::TiledGraphPortCapacity>
      residency_ports{};
  std::size_t residency_port_count{};
  std::uint32_t residency_graph_stage{residency::NoGraphStage};
  PipelineResidencyStage residency_stage{PipelineResidencyStage::Direct};
  PipelineResidencySemantic residency_semantic{PipelineResidencySemantic::None};
  std::array<PipelineResidencySemanticPort,
             residency::TiledGraphResourceCapacity>
      residency_semantic_ports{};
  std::size_t residency_semantic_port_count{};
  std::uint64_t residency_transfer_committed_bytes{};
  bool residency_transfer_prepared{};
  PipelineMemoryInventory memory_inventory;
  PipelinePlan plan{};
  mutable std::mutex gate;
  PipelinePhase phase{PipelinePhase::Ready};
  Stats stats{};
  ServiceFreeDirectProductEvidence service_free_direct{};
  CheckpointStats checkpoint_stats{};
  std::uint64_t status_entry_count{};
  std::size_t logical_step_count{};
  // One physical execution represents this many input-sealed evaluations at
  // one terminal observation after the cold temporal proof succeeds. It never
  // denotes this many public run()/generation transitions.
  std::uint32_t sealed_repetitions{1u};
  // Sealed during preparation. Warm accelerator execution uses this immutable
  // count instead of walking private Jobs to rediscover the active subset.
  std::uint32_t active_step_count{};
  PipelineAttemptState attempt{};
  std::uint64_t native_generation{};
  std::uint64_t observation_generation{};
  std::uint64_t observation_payload_epoch{};
  std::uint8_t native_parity{};
  std::uint8_t observation_parity{};
  Reason failure{Reason::Ok};
  // Explicit measurement epoch. It reuses PipelineStats as the sole public
  // evidence owner; these booleans only control whether terminals contribute
  // and whether an in-epoch observation/mutation invalidated the clean proof.
  enum class SampleState : std::uint8_t { Inactive, Clean, Dirty };
  SampleState samples{SampleState::Inactive};
  bool transactional{};
  bool preparing{true};
  bool observation_identity_valid{};
  // A submitted failure whose native generation control could not be rebased
  // cannot safely execute again: the next completion identity is unknowable.
  bool control_poisoned{};
  std::uint64_t frame_current{};
  std::uint64_t frame_peak{};
  std::uint64_t frame_bytes{};
  std::uint64_t frame_reused{};
  std::uint64_t frame_budget{};
  std::uint64_t transfer_peak{};
  std::uint64_t transfer_bytes{};
  std::uint64_t staging_peak{};
  std::uint64_t staging_bytes{};
  std::uint64_t staging_reused{};
  std::uint64_t staging_budget{};
  std::size_t unobserved_outputs{};
};

// Sole logical-ordinal -> physical-owner projection for transactional parity.
// Publication identities and fingerprints always retain the canonical
// ordinal; only the selected owner changes.
[[nodiscard]] inline const PipelineResource *
selected_pipeline_resource(const std::span<const PipelineResource> resources,
                           const std::uint32_t canonical_ordinal,
                           const bool alternate) noexcept {
  if (canonical_ordinal >= resources.size()) {
    return nullptr;
  }
  const PipelineResource &canonical = resources[canonical_ordinal];
  const std::uint32_t partner = canonical.partner;
  if (partner != PipelineResource::no_output &&
      (partner >= resources.size() ||
       resources[partner].partner != canonical_ordinal)) {
    return nullptr;
  }
  if (!alternate || partner == PipelineResource::no_output) {
    return &canonical;
  }
  return &resources[partner];
}

[[nodiscard]] inline const PipelineResource *
selected_pipeline_resource(const PipelineState &state,
                           const std::uint32_t ordinal,
                           const bool alternate) noexcept {
  return selected_pipeline_resource(state.resources, ordinal, alternate);
}

} // namespace rund::compute::detail
