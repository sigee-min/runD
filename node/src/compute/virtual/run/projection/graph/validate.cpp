#include "../internal.hpp"
#include "model.hpp"

#include "../../../../device/state.hpp"
#include "../../../backing.hpp"
#include <kernel/core/checked.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include <algorithm>
#include <limits>
#include <span>
#include <vector>

namespace rund::compute::detail {

bool validate_virtual_graph_projection(VirtualPipelineState &state,
                                       GraphProjectionInputs &inputs) noexcept {
  const VirtualBufferState *const input = virtual_input(state);
  PipelineState &prefix = *state.pipeline;
  const VirtualGeometry &geometry = state.geometry;
  const bool graph_reduction = geometry.route == VirtualRoute::GraphReduction;
  const bool graph_pointwise = geometry.route == VirtualRoute::GraphPointwise;
  residency::Pool *const pool = prefix.residency_pool.get();
  if (input == nullptr || state.output == nullptr || pool == nullptr ||
      prefix.residency == nullptr || state.input_count == 0u ||
      state.input_count != state.graph_input_resource_count ||
      state.input_count > residency::TiledGraphResourceCapacity - 2u ||
      !prefix.residency->graph_tiled() ||
      prefix.residency_stage != PipelineResidencyStage::Graph ||
      prefix.residency_graph_stage != 0u ||
      (!graph_reduction && !graph_pointwise) ||
      graph_terminal_pipeline(state, 0u) == nullptr ||
      graph_terminal_pipeline(state, 1u) == nullptr) {
    return false;
  }

  const residency::TiledGraphPlan &capacity_plan =
      prefix.residency->tiled_graph();
  const std::span<const residency::TiledGraphStage> stages =
      capacity_plan.stages();
  const auto output_port =
      stages.empty()
          ? std::vector<residency::TiledGraphPort>::const_iterator{}
          : std::find_if(stages.back().ports.begin(), stages.back().ports.end(),
                         [](const residency::TiledGraphPort port) {
                           return port.access == residency::Access::Write;
                         });
  const auto intermediate_plan_resource = std::find_if(
      capacity_plan.resources().begin(), capacity_plan.resources().end(),
      [](const residency::TiledGraphResource resource) {
        return resource.kind == residency::GraphResourceKind::Internal;
      });
  if (stages.size() < 2u || stages.front().ports.empty() ||
      stages.back().ports.size() < 2u ||
      output_port == stages.back().ports.end() ||
      std::count_if(stages.back().ports.begin(), stages.back().ports.end(),
                    [](const residency::TiledGraphPort port) {
                      return port.access == residency::Access::Write;
                    }) != 1 ||
      std::none_of(stages.back().ports.begin(), stages.back().ports.end(),
                   [](const residency::TiledGraphPort port) {
                     return port.access == residency::Access::Read;
                   }) ||
      intermediate_plan_resource == capacity_plan.resources().end() ||
      stages.back().domain != (graph_reduction
                                   ? residency::StageDomain::TilePartial
                                   : residency::StageDomain::Tile)) {
    return false;
  }

  const std::uint32_t input_resource = state.graph_input_resources[0u];
  const std::uint32_t intermediate_resource =
      intermediate_plan_resource->resource;
  const std::uint32_t output_resource = output_port->resource;
  const residency::TiledGraphResource *const input_plan_resource =
      capacity_plan.resource(input_resource);
  const residency::TiledGraphResource *const output_plan_resource =
      capacity_plan.resource(output_resource);
  const residency::PoolPhysicalOwner *const input_owner =
      input_plan_resource == nullptr
          ? nullptr
          : pool->graph_owner(input_plan_resource->physical_id);
  const residency::PoolPhysicalOwner *const intermediate_owner =
      pool->graph_owner(intermediate_plan_resource->physical_id);
  const residency::PoolPhysicalOwner *const output_owner =
      output_plan_resource == nullptr
          ? nullptr
          : pool->graph_owner(output_plan_resource->physical_id);
  for (std::size_t index = 0u; index < state.input_count; ++index) {
    const VirtualBufferState *const candidate = virtual_input(state, index);
    const residency::TiledGraphResource *const resource =
        capacity_plan.resource(state.graph_input_resources[index]);
    const residency::PoolPhysicalOwner *const owner =
        resource == nullptr ? nullptr
                            : pool->graph_owner(resource->physical_id);
    if (candidate == nullptr || candidate->backing == nullptr ||
        candidate->type != input->type || candidate->format != input->format ||
        candidate->count != input->count ||
        candidate->element_bytes != input->element_bytes ||
        candidate->bytes != input->bytes ||
        candidate->backing->size_bytes() != candidate->bytes ||
        resource == nullptr ||
        resource->kind != residency::GraphResourceKind::ExternalInput ||
        resource->persistence != residency::ResourcePersistence::Backing ||
        resource->page_bytes != pool->layout.input_page_bytes ||
        owner == nullptr ||
        !virtual_valid_regions(
            *pool, owner->bank_regions,
            prefix.device->backend == Backend::Cpu
                ? residency::FrameTier::Host
                : residency::FrameTier::Device,
            residency::FrameRole::Input,
            static_cast<std::uint32_t>(capacity_plan.frame_capacity()))) {
      return false;
    }
    const bool read = std::any_of(
        stages.begin(), stages.end(),
        [resource_id = resource->resource](
            const residency::TiledGraphStage &candidate_stage) {
          return std::any_of(
              candidate_stage.ports.begin(), candidate_stage.ports.end(),
              [resource_id](const residency::TiledGraphPort candidate_port) {
                return candidate_port.resource == resource_id &&
                       residency::reads(candidate_port.access);
              });
        });
    if (!read) {
      return false;
    }
  }

  const std::uint64_t capacity = capacity_plan.frame_capacity();
  const std::uint64_t input_page_bytes = pool->layout.input_page_bytes;
  const std::uint64_t intermediate_page_bytes =
      pool->layout.intermediate_page_bytes;
  const std::uint64_t control_page_bytes = pool->layout.control_page_bytes;
  const std::uint64_t output_page_bytes = pool->layout.output_page_bytes;
  std::uint64_t expected_output_page_bytes = 0u;
  std::uint64_t input_arena_bytes = 0u;
  std::uint64_t intermediate_arena_bytes = 0u;
  std::uint64_t control_arena_bytes = 0u;
  std::uint64_t output_arena_bytes = 0u;
  std::uint64_t host_input_group_bytes = 0u;
  std::uint64_t host_input_arena_bytes = 0u;
  std::uint64_t host_output_arena_bytes = 0u;
  std::uint64_t expected_host_storage = 0u;
  if (!kernel::checked::mul(geometry.output_frame_elements,
                            state.output->element_bytes,
                            expected_output_page_bytes) ||
      capacity == 0u || capacity > PipelineLeafCapacity ||
      input_page_bytes == 0u || intermediate_page_bytes == 0u ||
      control_page_bytes != sizeof(std::uint64_t) ||
      output_page_bytes != expected_output_page_bytes ||
      geometry.input_payload_elements == 0u ||
      geometry.input_frame_elements != geometry.input_payload_elements ||
      geometry.intermediate_frame_elements != geometry.input_frame_elements ||
      (graph_reduction && geometry.output_frame_elements != 1u) ||
      (graph_pointwise &&
       geometry.output_frame_elements != geometry.input_frame_elements) ||
      !kernel::checked::mul(capacity, input_page_bytes, input_arena_bytes) ||
      !kernel::checked::mul(capacity, intermediate_page_bytes,
                            intermediate_arena_bytes) ||
      !kernel::checked::mul(capacity, control_page_bytes,
                            control_arena_bytes) ||
      !kernel::checked::mul(capacity, output_page_bytes, output_arena_bytes) ||
      !kernel::checked::mul(pool->layout.host_frame_capacity, input_page_bytes,
                            host_input_group_bytes) ||
      !kernel::checked::mul(host_input_group_bytes,
                            pool->layout.graph_host_input_count,
                            host_input_arena_bytes) ||
      !kernel::checked::mul(pool->layout.host_output_frame_capacity,
                            output_page_bytes, host_output_arena_bytes) ||
      (prefix.device->backend != Backend::Cpu &&
       (!kernel::checked::add(host_input_arena_bytes, host_output_arena_bytes,
                              expected_host_storage) ||
        !kernel::checked::mul(expected_host_storage, residency::Pool::BankCount,
                              expected_host_storage))) ||
      pool->layout.frame_capacity != capacity ||
      pool->layout.graph_host_input_count != state.input_count ||
      pool->layout.host_frame_capacity < capacity ||
      pool->layout.host_output_frame_capacity < capacity ||
      pool->host_storage_bytes != expected_host_storage ||
      (prefix.device->backend == Backend::Cpu
           ? pool->host_input_frame_count != 0u
           : pool->host_input_frame_count !=
                 pool->layout.host_frame_capacity *
                     pool->layout.graph_host_input_count *
                     residency::Pool::BankCount) ||
      input_owner == nullptr || intermediate_owner == nullptr ||
      output_owner == nullptr ||
      !virtual_valid_regions(
          *pool, input_owner->bank_regions,
          prefix.device->backend == Backend::Cpu ? residency::FrameTier::Host
                                                 : residency::FrameTier::Device,
          residency::FrameRole::Input, static_cast<std::uint32_t>(capacity)) ||
      !virtual_valid_regions(*pool, intermediate_owner->bank_regions,
                             prefix.device->backend == Backend::Cpu
                                 ? residency::FrameTier::Host
                                 : residency::FrameTier::Device,
                             residency::FrameRole::Intermediate,
                             static_cast<std::uint32_t>(capacity)) ||
      !virtual_valid_regions(
          *pool, output_owner->bank_regions,
          prefix.device->backend == Backend::Cpu ? residency::FrameTier::Host
                                                 : residency::FrameTier::Device,
          residency::FrameRole::Output, static_cast<std::uint32_t>(capacity)) ||
      input->backing->size_bytes() != input->bytes ||
      state.output->backing->size_bytes() != state.output->bytes) {
    return false;
  }

  inputs = GraphProjectionInputs{
      .input = input,
      .prefix = &prefix,
      .pool = pool,
      .capacity_plan = &capacity_plan,
      .input_owner = input_owner,
      .intermediate_owner = intermediate_owner,
      .output_owner = output_owner,
      .input_resource = input_resource,
      .intermediate_resource = intermediate_resource,
      .output_resource = output_resource,
      .capacity = capacity,
      .input_page_bytes = input_page_bytes,
      .intermediate_page_bytes = intermediate_page_bytes,
      .control_page_bytes = control_page_bytes,
      .output_page_bytes = output_page_bytes,
      .input_arena_bytes = input_arena_bytes,
      .intermediate_arena_bytes = intermediate_arena_bytes,
      .control_arena_bytes = control_arena_bytes,
      .output_arena_bytes = output_arena_bytes,
      .host_input_arena_bytes = host_input_arena_bytes,
      .host_output_arena_bytes = host_output_arena_bytes,
      .graph_reduction = graph_reduction,
      .graph_pointwise = graph_pointwise,
  };
  return true;
}

} // namespace rund::compute::detail
