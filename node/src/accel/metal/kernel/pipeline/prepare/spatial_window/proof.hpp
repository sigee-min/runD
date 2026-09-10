#pragma once

#include "equality.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

struct MetalSpatialWindowLocalProof final {
  const KernelExecution *execution{};
  rund::kernel::ResidentBufferRef input{};
  rund::kernel::ResidentBufferRef output{};
  const void *input_handle{};
  const void *output_handle{};
  const void *bound_step{};
  const void *producer_step{};
  const void *consumer_step{};
  const void *binding_owner{};
  std::uint64_t input_index{};
  std::uint64_t output_index{};
  std::uint64_t input_binding_index{};
  std::uint64_t output_binding_index{};
  std::uint64_t range_begin{};
  std::uint64_t range_count{};
  std::uint64_t range_dispatch_count{};
  std::array<rund::kernel::ResidentBufferRef, 6u> edge_refs{};
  std::array<const void *, 6u> edge_handles{};
  std::array<const void *, 3u> edge_steps{};
  bool edge_bindings_captured{};
  bool captured{};

  [[nodiscard]] bool valid() const noexcept {
    return captured && execution != nullptr && bound_step != nullptr &&
           producer_step != nullptr && consumer_step != nullptr &&
           binding_owner != nullptr && edge_bindings_captured &&
           input_handle != nullptr && output_handle != nullptr &&
           input.id != 0u && output.id != 0u && input.bytes != 0u &&
           output.bytes != 0u && input.count != 0u && output.count != 0u &&
           input.element_bytes != 0u && output.element_bytes != 0u &&
           input.stride_bytes >= input.element_bytes &&
           output.stride_bytes >= output.element_bytes &&
           input.offset_bytes <= input.bytes &&
           output.offset_bytes <= output.bytes &&
           input.usage == rund::kernel::kResidentUsageRead &&
           output.usage == rund::kernel::kResidentUsageWrite &&
           MetalSpatialWindowResidentRefValid(edge_refs[0u]) &&
           MetalSpatialWindowResidentRefValid(edge_refs[1u]) &&
           MetalSpatialWindowResidentRefValid(edge_refs[2u]) &&
           MetalSpatialWindowResidentRefValid(edge_refs[3u]) &&
           MetalSpatialWindowResidentRefValid(edge_refs[4u]) &&
           MetalSpatialWindowResidentRefValid(edge_refs[5u]) &&
           edge_refs[0u].usage == rund::kernel::kResidentUsageRead &&
           edge_refs[1u].usage == rund::kernel::kResidentUsageWrite &&
           edge_refs[2u].usage == rund::kernel::kResidentUsageRead &&
           edge_refs[3u].usage == rund::kernel::kResidentUsageWrite &&
           edge_refs[4u].usage == rund::kernel::kResidentUsageRead &&
           edge_refs[5u].usage == rund::kernel::kResidentUsageWrite &&
           edge_handles[0u] != nullptr && edge_handles[1u] != nullptr &&
           edge_handles[2u] != nullptr && edge_handles[3u] != nullptr &&
           edge_handles[4u] != nullptr && edge_handles[5u] != nullptr &&
           edge_steps[0u] != nullptr && edge_steps[1u] != nullptr &&
           edge_steps[2u] != nullptr;
  }
};

// Backend-private proof for the stateless spatial Window route. This is
// deliberately separate from MetalWindow, which describes temporal
// recurrence binding. The proof retains exact graph edges, Window descriptor
// and plan fields, range identities, and every prepared local RangeBinds row;
// it carries no ResidentState or native recurrence storage.
struct MetalSpatialWindowProof final {
  const KernelExecution *execution{};
  std::uint32_t local_count{};
  std::uint32_t step_count{};
  std::uint32_t producer_index{};
  std::uint32_t window{};
  std::uint32_t consumer_index{};
  std::array<std::uint64_t, 6u> binding_indices{};
  std::uint64_t window_input_binding{};
  std::uint64_t window_output_binding{};
  std::array<std::uint64_t, 11u> descriptor{};
  std::array<std::uint64_t, 17u> plan{};
  std::array<std::uint64_t, 12u> range{};
  RangeIdentity source_identity{};
  RangeIdentity execution_identity{};
  RangeIdentity candidate_identity{};
  RangePath candidate_path{RangePath::Direct};
  std::uint8_t candidate_disposition{};
  std::uint32_t candidate_width{};
  std::uint32_t candidate_radius{};
  std::uint32_t halo_radius{};
  std::uint64_t halo_frame{};
  std::uint64_t halo_payload{};
  std::uint8_t halo_boundary{};
  std::uint32_t stage_count{};
  std::uint32_t temporary_count{};
  std::uint8_t source_variant{};
  // Cold-sized only after the spatial route is proved. Ordinary sequences
  // retain no local rows; the global step limit is not a storage request.
  std::vector<MetalSpatialWindowLocalProof> locals{};
  bool window_present{};
  bool candidate_captured{};
  bool halo_captured{};
  bool local_bindings_complete{};
  bool admitted{};

  [[nodiscard]] bool matches(const BackendRun &run) const noexcept {
    return admitted && execution != nullptr && run.execution != nullptr &&
           MetalSpatialWindowSameKernelExecution(*execution, *run.execution) &&
           run.steps != nullptr && run.step_count == step_count &&
           step_count == 3u && producer_index == 0u && window == 1u &&
           consumer_index == 2u;
  }

  [[nodiscard]] bool graph_bindings_valid() const noexcept {
    if (execution == nullptr ||
        execution->graph_roles.size() !=
            execution->graph_alias_representatives.size()) {
      return false;
    }
    constexpr std::array<rund::kernel::BufferRole, 6u> expected_roles{
        rund::kernel::BufferRole::Read, rund::kernel::BufferRole::Write,
        rund::kernel::BufferRole::Read, rund::kernel::BufferRole::Write,
        rund::kernel::BufferRole::Read, rund::kernel::BufferRole::Write};
    std::array<std::uint64_t, 6u> representatives{};
    for (std::size_t edge = 0u; edge < binding_indices.size(); ++edge) {
      const std::uint64_t binding = binding_indices[edge];
      if (binding >= execution->graph_roles.size() ||
          execution->graph_roles[static_cast<std::size_t>(binding)] !=
              expected_roles[edge]) {
        return false;
      }
      const std::uint64_t representative =
          execution
              ->graph_alias_representatives[static_cast<std::size_t>(binding)];
      if (representative >= execution->graph_roles.size()) {
        return false;
      }
      representatives[edge] = representative;
    }
    return representatives[1u] == representatives[2u] &&
           representatives[3u] == representatives[4u];
  }

  [[nodiscard]] bool shape_valid() const noexcept {
    return admitted && step_count == 3u && producer_index == 0u &&
           window == 1u && consumer_index == 2u && local_count != 0u &&
           local_count <= PreparedPipelineStepCapacity &&
           locals.size() == local_count &&
           descriptor[2u] == static_cast<std::uint64_t>(
                                 rund::kernel::WindowBoundary::Clamp) &&
           descriptor[5u] ==
               static_cast<std::uint64_t>(
                   rund::kernel::ComputeCountSource::Descriptor) &&
           descriptor[9u] == 1u && descriptor[6u] == descriptor[7u] &&
           descriptor[6u] != 0u && plan[2u] == descriptor[2u] &&
           plan[5u] == descriptor[5u] && plan[6u] == descriptor[6u] &&
           plan[7u] == descriptor[7u] && plan[8u] == descriptor[8u] &&
           plan[9u] == descriptor[9u] && plan[10u] == descriptor[10u] &&
           plan[16u] != 0u && candidate_captured &&
           candidate_path == RangePath::SharedHalo &&
           candidate_disposition == static_cast<std::uint8_t>(candidate_path) &&
           candidate_identity == source_identity && candidate_width != 0u &&
           candidate_radius != 0u && halo_captured &&
           halo_radius == candidate_radius &&
           halo_frame ==
               static_cast<std::uint64_t>(candidate_width) +
                   2u * static_cast<std::uint64_t>(candidate_radius) &&
           halo_payload == halo_frame * range[6u] &&
           halo_boundary == range[0u] && stage_count == 1u &&
           range[0u] == static_cast<std::uint64_t>(RangeBoundary::Clamp) &&
           range[1u] == range[2u] && range[4u] == 1u && range[11u] != 0u &&
           source_identity.hi != 0u && source_identity.lo != 0u &&
           execution_identity.hi != 0u && execution_identity.lo != 0u &&
           graph_bindings_valid() &&
           binding_indices[0u] != binding_indices[1u] &&
           binding_indices[2u] != binding_indices[3u] &&
           binding_indices[4u] != binding_indices[5u];
  }

  [[nodiscard]] bool local_valid(const std::size_t local) const noexcept {
    return shape_valid() && local_bindings_complete && local < local_count &&
           locals[local].valid() &&
           locals[local].input_index == binding_indices[2u] &&
           locals[local].output_index == binding_indices[3u] &&
           locals[local].input_binding_index == binding_indices[2u] &&
           locals[local].output_binding_index == binding_indices[3u];
  }

  [[nodiscard]] bool operation_valid() const noexcept {
    if (execution == nullptr || execution->steps.size() != step_count ||
        producer_index >= execution->steps.size() ||
        window >= execution->steps.size() ||
        consumer_index >= execution->steps.size() ||
        execution->steps[producer_index].kind() !=
            rund::kernel::NodeKind::Map ||
        execution->steps[window].kind() != rund::kernel::NodeKind::Window ||
        execution->steps[consumer_index].kind() !=
            rund::kernel::NodeKind::Map) {
      return false;
    }
    const operation::Window &window_operation =
        execution->steps[window].operation.get<operation::Window>();
    if (!window_operation.range.ok()) {
      return false;
    }
    const RangeCandidate &candidate = window_operation.range.candidate();
    const RangeShape &shape = window_operation.range.shape();
    return descriptor == MetalSpatialWindowDescriptorKey(window_operation) &&
           plan == MetalSpatialWindowPlanKey(window_operation) &&
           range == MetalSpatialWindowRangeKey(window_operation.range) &&
           candidate_captured && candidate_path == candidate.disposition() &&
           candidate_disposition ==
               static_cast<std::uint8_t>(candidate.disposition()) &&
           candidate_identity == window_operation.range.source_identity() &&
           candidate_width == candidate.width() &&
           candidate_radius == candidate.radius_capacity() && halo_captured &&
           halo_radius == candidate.radius_capacity() &&
           halo_frame == MetalSpatialWindowHaloFrame(candidate) &&
           halo_payload == MetalSpatialWindowHaloPayload(candidate, shape) &&
           halo_boundary == static_cast<std::uint8_t>(shape.boundary());
  }

  [[nodiscard]] bool local_matches(const std::size_t local) const noexcept {
    if (!local_valid(local) || execution == nullptr || local >= local_count) {
      return false;
    }
    const MetalSpatialWindowLocalProof &row = locals[local];
    const KernelExecution *const row_execution = row.execution;
    const BoundStep *const producer =
        static_cast<const BoundStep *>(row.producer_step);
    const BoundStep *const window_step =
        static_cast<const BoundStep *>(row.bound_step);
    const BoundStep *const consumer =
        static_cast<const BoundStep *>(row.consumer_step);
    if (row_execution == nullptr ||
        !MetalSpatialWindowSameKernelExecution(*execution, *row_execution) ||
        producer == nullptr || window_step == nullptr || consumer == nullptr ||
        row.edge_steps[0u] != producer || row.edge_steps[1u] != window_step ||
        row.edge_steps[2u] != consumer || producer->index != producer_index ||
        window_step->index != window || consumer->index != consumer_index ||
        producer->step != &row_execution->steps[producer_index] ||
        window_step->step != &row_execution->steps[window] ||
        consumer->step != &row_execution->steps[consumer_index] ||
        !MetalSpatialWindowPlannedStepValid(
            *producer, row_execution->steps[producer_index]) ||
        !MetalSpatialWindowPlannedStepValid(*window_step,
                                            row_execution->steps[window]) ||
        !MetalSpatialWindowPlannedStepValid(
            *consumer, row_execution->steps[consumer_index]) ||
        producer->source_binds == nullptr ||
        window_step->source_binds == nullptr ||
        consumer->source_binds == nullptr || producer->control.active() ||
        window_step->control.active() || consumer->control.active() ||
        !producer->resets.empty() || !window_step->resets.empty() ||
        !consumer->resets.empty() ||
        !BoundStepMatches(*producer, rund::kernel::NodeKind::Map) ||
        !BoundStepMatches(*window_step, rund::kernel::NodeKind::Window) ||
        !BoundStepMatches(*consumer, rund::kernel::NodeKind::Map) ||
        row.binding_owner != window_step->source_binds ||
        producer->source_binds != row.binding_owner ||
        consumer->source_binds != row.binding_owner) {
      return false;
    }
    const StepBinds *const producer_binds =
        BindingsFor<StepBinds>(*producer, rund::kernel::NodeKind::Map);
    const RangeBinds *const window_binds =
        BindingsFor<RangeBinds>(*window_step, rund::kernel::NodeKind::Window);
    const StepBinds *const consumer_binds =
        BindingsFor<StepBinds>(*consumer, rund::kernel::NodeKind::Map);
    if (producer_binds == nullptr || window_binds == nullptr ||
        consumer_binds == nullptr || !producer_binds->valid() ||
        !consumer_binds->valid() || producer_binds->inputs.size() != 1u ||
        producer_binds->outputs.size() != 1u ||
        consumer_binds->inputs.size() != 1u ||
        consumer_binds->outputs.size() != 1u ||
        window_binds->input == nullptr ||
        window_binds->input_handle == nullptr ||
        window_binds->output == nullptr ||
        window_binds->output_handle == nullptr ||
        *window_binds->input_handle == nullptr ||
        *window_binds->output_handle == nullptr) {
      return false;
    }
    const rund::kernel::ResidentBindingRange producer_inputs =
        producer_binds->inputs.range();
    const rund::kernel::ResidentBindingRange producer_outputs =
        producer_binds->outputs.range();
    const rund::kernel::ResidentBindingRange consumer_inputs =
        consumer_binds->inputs.range();
    const rund::kernel::ResidentBindingRange consumer_outputs =
        consumer_binds->outputs.range();
    const std::array<const rund::kernel::ResidentBindingRange *, 4u> views{
        &producer_inputs, &producer_outputs, &consumer_inputs,
        &consumer_outputs};
    const std::array<std::size_t, 4u> edge_positions{0u, 1u, 4u, 5u};
    for (std::size_t index = 0u; index < views.size(); ++index) {
      const rund::kernel::ResidentBufferRef *const ref = views[index]->ref(0u);
      const std::shared_ptr<void> *const handle = views[index]->handle(0u);
      if (ref == nullptr || handle == nullptr || *handle == nullptr ||
          !MetalSpatialWindowResidentRefEqual(
              *ref, row.edge_refs[edge_positions[index]]) ||
          handle->get() != row.edge_handles[edge_positions[index]]) {
        return false;
      }
    }
    if (!MetalSpatialWindowResidentRefEqual(*window_binds->input,
                                            row.edge_refs[2u]) ||
        !MetalSpatialWindowResidentRefEqual(*window_binds->output,
                                            row.edge_refs[3u]) ||
        window_binds->input_handle->get() != row.edge_handles[2u] ||
        window_binds->output_handle->get() != row.edge_handles[3u]) {
      return false;
    }
    if (!MetalSpatialWindowResidentRefPhysicalEqual(row.edge_refs[1u],
                                                    row.edge_refs[2u]) ||
        !MetalSpatialWindowResidentRefPhysicalEqual(row.edge_refs[3u],
                                                    row.edge_refs[4u]) ||
        row.edge_handles[1u] != row.edge_handles[2u] ||
        row.edge_handles[3u] != row.edge_handles[4u]) {
      return false;
    }
    const RunBinds *const owner = window_step->source_binds;
    if (owner->refs() == nullptr || owner->handles() == nullptr ||
        row.input_binding_index >= owner->size() ||
        row.output_binding_index >= owner->size() ||
        &owner->refs()[row.input_binding_index] != window_binds->input ||
        &owner->refs()[row.output_binding_index] != window_binds->output ||
        &owner->handles()[row.input_binding_index] !=
            window_binds->input_handle ||
        &owner->handles()[row.output_binding_index] !=
            window_binds->output_handle) {
      return false;
    }
    return true;
  }
};

struct BackendBatchEntry;
struct BackendPublish;
struct MapRecurrence;
struct NestedAggregate;
struct TileTransducer;

[[nodiscard]] MetalSpatialWindowProof ProveMetalSpatialWindow(
    const std::span<const BackendBatchEntry> entries,
    const MapRecurrence &recurrence,
    const std::span<const TileTransducer> transducers,
    const std::span<const NestedAggregate> aggregates,
    const std::span<const BackendPublish> publications,
    const std::uint32_t local_count);
#endif

} // namespace rund::node::accel::detail
