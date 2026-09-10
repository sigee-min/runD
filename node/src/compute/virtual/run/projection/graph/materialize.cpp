#include "../identity.hpp"
#include "model.hpp"

#include "../../../backing.hpp"

#include <algorithm>

namespace rund::compute::detail {

bool materialize_virtual_graph_projection(
    VirtualPipelineState &state, const GraphProjectionInputs &inputs,
    const GraphProjectionBanks &banks, const VirtualActiveProjection &active,
    VirtualRunProjection &projection) noexcept {
  std::array<VirtualRunInputProjection, VirtualPipelineState::InputCapacity>
      run_inputs{};
  std::array<std::uint64_t, VirtualPipelineState::InputCapacity> input_ids{};
  std::array<std::uint64_t, VirtualPipelineState::InputCapacity>
      input_versions{};
  const VirtualGeometry &geometry = state.geometry;
  const std::uint64_t output_id =
      VirtualBackingAccess::id(*state.output->backing);
  const std::uint64_t output_version =
      VirtualBackingAccess::version(*state.output->backing);
  const bool partial =
      active.active_count % geometry.input_payload_elements != 0u;
  const std::uint64_t boundary_extent = partial ? active.active_count : 0u;
  constexpr std::uint64_t kIntermediateDomain = 0x67726170682e6d64ull;
  constexpr std::uint64_t kOutputDomain = 0x67726170682e6f75ull;
  const residency::Identity graph_identity{.hi = geometry.materialization_hi,
                                           .lo = geometry.materialization_lo};
  const residency::Identity input_identity = input_materialization_identity(
      inputs.input->type, inputs.input->format, inputs.input_page_bytes,
      inputs.input_page_bytes, 0u, geometry.input_frame_elements, 0u);
  for (std::size_t index = 0u; index < state.input_count; ++index) {
    const VirtualBufferState *const candidate = virtual_input(state, index);
    input_ids[index] = VirtualBackingAccess::id(*candidate->backing);
    input_versions[index] = VirtualBackingAccess::version(*candidate->backing);
    if (input_ids[index] == 0u) {
      return false;
    }
    run_inputs[index] = VirtualRunInputProjection{
        .capacity_bytes = candidate->bytes,
        .visible_bytes = active.input_bytes,
        .identity_hi = input_identity.hi,
        .identity_lo = input_identity.lo,
        .backing = input_ids[index],
        .version = input_versions[index],
        .type = candidate->type,
    };
  }

  const std::uint64_t input_id = input_ids[0u];
  const std::uint64_t input_version = input_versions[0u];
  const residency::Identity intermediate_identity =
      transient_materialization_identity(
          graph_identity, inputs.intermediate_resource, kIntermediateDomain);
  const residency::Identity output_identity =
      inputs.graph_pointwise
          ? output_materialization_identity(
                state.output->type, state.output->format,
                inputs.output_page_bytes, geometry.output_frame_elements)
          : transient_materialization_identity(
                graph_identity, inputs.output_resource, kOutputDomain);
  const residency::GraphMaterialization input_materialization{
      .resource = inputs.input_resource,
      .key = {.backing = input_id,
              .version = input_version,
              .materialization_hi = input_identity.hi,
              .materialization_lo = input_identity.lo},
      .page_bytes = inputs.input_page_bytes,
      .page_count = active.graph.page_count(),
      .boundary_extent = boundary_extent,
  };
  const residency::GraphMaterialization intermediate_materialization{
      .resource = inputs.intermediate_resource,
      .key = {.backing = input_id,
              .version = input_version,
              .materialization_hi = intermediate_identity.hi,
              .materialization_lo = intermediate_identity.lo,
              .domain = residency::CacheDomain::Transient},
      .page_bytes = inputs.intermediate_page_bytes,
      .page_count = active.graph.page_count(),
      .boundary_extent = boundary_extent,
  };
  const residency::GraphMaterialization output_materialization{
      .resource = inputs.output_resource,
      .key = {.backing = output_id,
              .version = output_version,
              .materialization_hi = output_identity.hi,
              .materialization_lo = output_identity.lo,
              .domain = inputs.graph_pointwise
                            ? residency::CacheDomain::Backing
                            : residency::CacheDomain::Transient},
      .page_bytes = inputs.output_page_bytes,
      .page_count = active.graph.page_count(),
      .boundary_extent = boundary_extent,
  };
  if (input_materialization.resource == 0u ||
      intermediate_materialization.resource == 0u ||
      output_materialization.resource == 0u || input_id == 0u ||
      output_id == 0u) {
    return false;
  }

  std::array<std::uint32_t, residency::TiledGraphResourceCapacity>
      graph_resources{};
  std::array<residency::GraphMaterialization,
             residency::TiledGraphResourceCapacity>
      graph_materializations{};
  if (inputs.capacity_plan->resources().empty() ||
      inputs.capacity_plan->resources().size() > graph_resources.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < inputs.capacity_plan->resources().size();
       ++index) {
    const residency::TiledGraphResource &resource =
        inputs.capacity_plan->resources()[index];
    const bool backing =
        resource.persistence == residency::ResourcePersistence::Backing;
    const bool output =
        resource.kind == residency::GraphResourceKind::ExternalOutput;
    const auto input_resource =
        std::find(state.graph_input_resources.begin(),
                  state.graph_input_resources.begin() + state.input_count,
                  resource.resource);
    const std::size_t input_index = static_cast<std::size_t>(
        std::distance(state.graph_input_resources.begin(), input_resource));
    const bool backing_input = backing && !output;
    const bool backing_output = backing && output;
    if ((backing_input &&
         (input_resource ==
              state.graph_input_resources.begin() + state.input_count ||
          input_index >= state.input_count)) ||
        (backing_output && !inputs.graph_pointwise)) {
      return false;
    }
    const residency::Identity identity =
        backing_input    ? input_identity
        : backing_output ? output_identity
                         : transient_materialization_identity(
                               graph_identity, resource.resource,
                               output ? kOutputDomain : kIntermediateDomain);
    graph_resources[index] = resource.resource;
    graph_materializations[index] = residency::GraphMaterialization{
        .resource = resource.resource,
        .key = {.backing = backing_input ? input_ids[input_index]
                           : output      ? output_id
                                         : input_id,
                .version = backing_input ? input_versions[input_index]
                           : output      ? output_version
                                         : input_version,
                .materialization_hi = identity.hi,
                .materialization_lo = identity.lo,
                .domain = backing ? residency::CacheDomain::Backing
                                  : residency::CacheDomain::Transient},
        .page_bytes = resource.page_bytes,
        .page_count = active.graph.page_count(),
        .boundary_extent = boundary_extent,
    };
  }

  projection = VirtualRunProjection{
      .active = active,
      .inputs = run_inputs,
      .input_count = state.input_count,
      .input_capacity_bytes = inputs.input->bytes,
      .input_visible_bytes = active.input_bytes,
      .cache_extent = active.active_count,
      .input_identity_hi = input_identity.hi,
      .input_identity_lo = input_identity.lo,
      .result_identity_hi = geometry.materialization_hi,
      .result_identity_lo = geometry.materialization_lo,
      .input_backing = input_id,
      .input_version = input_version,
      .output_backing = output_id,
      .output_version = output_version,
      .input_page_bytes = inputs.input_page_bytes,
      .intermediate_page_bytes = inputs.intermediate_page_bytes,
      .control_page_bytes = inputs.control_page_bytes,
      .output_page_bytes = inputs.output_page_bytes,
      .input_payload_bytes = inputs.input_page_bytes,
      .output_payload_bytes = inputs.output_page_bytes,
      .input_frame_elements = geometry.input_frame_elements,
      .device_vsm_required = geometry.device_vsm_required,
      .reduction = inputs.graph_reduction,
      .graph_execution = true,
      .graph_reduction = inputs.graph_reduction,
      .input_type = inputs.input->type,
      .output_type = state.output->type,
      .operation = geometry.operation,
      .frame_capacity = inputs.capacity,
      .input_arena_bytes = inputs.input_arena_bytes,
      .intermediate_arena_bytes = inputs.intermediate_arena_bytes,
      .control_arena_bytes = inputs.control_arena_bytes,
      .output_arena_bytes = inputs.output_arena_bytes,
      .input_regions = inputs.input_owner->bank_regions,
      .intermediate_regions = inputs.intermediate_owner->bank_regions,
      .output_regions = inputs.output_owner->bank_regions,
      .first_intermediate_frame =
          inputs.intermediate_owner->bank_regions[0].first,
      .first_output_frame = inputs.output_owner->bank_regions[0].first,
      .first_host_input_frame = inputs.pool->first_host_input_frame,
      .host_input_count = inputs.pool->layout.graph_host_input_count,
      .host_frame_capacity = inputs.pool->layout.host_frame_capacity,
      .first_host_output_frame = inputs.pool->first_host_output_frame,
      .host_output_frame_capacity =
          inputs.prefix->device->backend == Backend::Cpu
              ? 0u
              : inputs.pool->layout.host_output_frame_capacity,
      .input_host_banks = banks.input,
      .control_host_banks = banks.control,
      .output_host_banks = banks.output,
      .graph_input = input_materialization,
      .graph_intermediate = intermediate_materialization,
      .graph_output = output_materialization,
      .graph_resources = graph_resources,
      .graph_materializations = graph_materializations,
      .graph_resource_count = inputs.capacity_plan->resources().size(),
  };
  return true;
}

} // namespace rund::compute::detail
