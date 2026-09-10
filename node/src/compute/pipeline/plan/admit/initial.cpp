#include "../../state/assembly.hpp"
#include "model.hpp"

#include "../../../device/residency/pool.hpp"
#include "../../../type.hpp"
#include "../../claim.hpp"
#include "../../local.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

namespace rund::compute::detail {

Status admit_initial(AdmissionDraft &draft) {
  const PipelineBuildState *const build = &draft.build;
  const PipelineMemoryPlan &plan = draft.plan;
  const PipelineBuildSnapshot &frozen = draft.frozen;
  const std::size_t resource_count = draft.resource_count;
  if (plan.window_states.size() != frozen.steps.size() ||
      plan.step_resources.size() != frozen.steps.size() ||
      frozen.commit != !plan.state_pair_resources.empty() ||
      build->materialized_resources.size() != plan.resources.size() ||
      plan.resources.size() != plan.hazards.lifetimes.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (resource_count > PipelineResourceCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }

  auto state = std::make_shared<PipelineState>();
  state->device = frozen.device;
  state->publication = std::make_shared<PipelinePublicationState>();
  state->publication->device = frozen.device;
  state->sealed_repetitions = frozen.sealed_repetitions;
  state->plan = plan.summary;
  if (plan.residency.pages != nullptr) {
    std::uint64_t residency_page_bytes = 0u;
    std::uint64_t host_input_bytes = 0u;
    std::uint64_t host_output_bytes = 0u;
    std::uint64_t expected_host_storage_bytes = 0u;
    const bool direct = plan.residency.stage == PipelineResidencyStage::Direct;
    const bool graph = plan.residency.stage == PipelineResidencyStage::Graph;
    const std::uint64_t frame_capacity =
        direct ? plan.residency.pages->stream().frame_capacity()
               : plan.residency.pages->tiled_graph().frame_capacity();
    if (!plan.residency.pages->identity() || plan.residency.pool == nullptr ||
        (direct != plan.residency.pages->streamed()) ||
        (graph != plan.residency.pages->graph_tiled()) ||
        plan.residency.input_page_bytes == 0u ||
        plan.residency.output_page_bytes == 0u ||
        (direct && !kernel::checked::add(plan.residency.input_page_bytes,
                                         plan.residency.output_page_bytes,
                                         residency_page_bytes)) ||
        (graph &&
         (!residency::graph_pool_footprint(
              *plan.residency.pool,
              plan.residency.pages->tiled_graph().physical_classes(),
              residency_page_bytes) ||
          residency_page_bytes != plan.residency.pages->page_bytes())) ||
        residency_page_bytes != plan.residency.pages->page_bytes() ||
        plan.residency.frame_count == 0u ||
        (state->device->backend != Backend::Cpu &&
         (!kernel::checked::mul(plan.residency.pool->layout.input_page_bytes,
                                plan.residency.pool->layout.host_frame_capacity,
                                host_input_bytes) ||
          !kernel::checked::mul(
              host_input_bytes,
              plan.residency.pool->layout.graph_host_input_count,
              host_input_bytes) ||
          !kernel::checked::mul(
              plan.residency.pool->layout.output_page_bytes,
              plan.residency.pool->layout.host_output_frame_capacity,
              host_output_bytes) ||
          !kernel::checked::add(host_input_bytes, host_output_bytes,
                                expected_host_storage_bytes) ||
          !kernel::checked::mul(expected_host_storage_bytes,
                                residency::Pool::BankCount,
                                expected_host_storage_bytes))) ||
        (state->device->backend != Backend::Cpu &&
         plan.residency.pool->host_storage == nullptr) ||
        plan.residency.pool->host_storage_bytes !=
            expected_host_storage_bytes ||
        plan.residency.frame_count != frame_capacity ||
        ((state->device->backend == Backend::Vulkan) !=
         (plan.residency.transfer_committed_bytes != 0u)) ||
        state->plan.residency.logical_bytes != plan.residency.logical_bytes ||
        state->plan.residency.resident_bytes != plan.residency.resident_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    state->residency = plan.residency.pages;
    state->residency_pool = plan.residency.pool;
    state->residency_transfer_committed_bytes =
        plan.residency.transfer_committed_bytes;
    state->residency_bank = plan.residency.bank;
    state->residency_stage = plan.residency.stage;
  }
  state->steps.resize(frozen.steps.size());
  if (plan.window_controls.size() > std::numeric_limits<std::uint16_t>::max()) {
    return Status::fail(Reason::PipelineCapacity);
  }
  state->windows.entries_.resize(plan.window_controls.size());
  draft.initialized_windows.assign(plan.window_controls.size(), false);
  state->resources.reserve(resource_count);
  state->barriers.resize(frozen.steps.size());
  if (frozen.profile == PipelineProfile::Steps) {
    state->profile = std::make_unique<PipelineProfileState>();
    state->profile->steps.resize(frozen.steps.size());
    state->profile->started_ns.resize(frozen.steps.size());
    state->profile->started.resize(frozen.steps.size());
  }

  for (std::size_t ordinal = 0u; ordinal < resource_count; ++ordinal) {
    const PipelineResolvedResourcePlan &planned = plan.resources[ordinal];
    const std::shared_ptr<BufferState> &owner =
        build->materialized_resources[ordinal];
    const bool owned =
        std::holds_alternative<PipelineInternalResourcePlan>(planned.locator);
    if (owner == nullptr ||
        planned.count > std::numeric_limits<std::size_t>::max() ||
        planned.bytes > std::numeric_limits<std::size_t>::max() ||
        (planned.first_write != resource::NoNode &&
         planned.first_write >= frozen.steps.size())) {
      return Status::fail(owner == nullptr ? Reason::BindingInvalid
                                           : Reason::PipelineInvalid);
    }
    state->resources.push_back(PipelineResource{
        .buffer = owner,
        .type = planned.type,
        .format = planned.format,
        .count = static_cast<std::size_t>(planned.count),
        .bytes = static_cast<std::size_t>(planned.bytes),
        .output = PipelineResource::no_output,
        .first_write = planned.first_write,
        .owned = owned,
        .terminal_publish = false,
    });
  }

  draft.state = std::move(state);
  return Status::success();
}

} // namespace rund::compute::detail
