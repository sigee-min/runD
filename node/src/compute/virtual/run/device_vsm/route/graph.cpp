#include "internal.hpp"

#include "../../backing_set.hpp"
#include "../internal.hpp"
#include "../operations.hpp"
#include "../staging/cohort.hpp"

#include "../../../../backend.hpp"
#include "../../../backing.hpp"
#include "../../../graph/reduce/wavefront.hpp"

#include <algorithm>
#include <array>

namespace rund::compute::detail::device_vsm_route_detail {

bool graph_host_product_eligible(const VirtualPipelineState &state,
                                 const std::span<VirtualBacking *const> inputs,
                                 const VirtualRunProjection &run) noexcept {
  if (!run.graph_execution() || run.graph_reduction() ||
      state.geometry.route != VirtualRoute::GraphPointwise ||
      state.pipeline == nullptr || state.pipeline->device == nullptr ||
      state.pipeline->device->backend == Backend::Cpu ||
      state.pipeline->residency == nullptr ||
      !state.pipeline->residency->graph_tiled() || !run.active.graph.valid() ||
      run.active.graph.batch_count() < 2u ||
      run.active.graph.frame_capacity() == 0u ||
      run.active.graph.prefetch_distance() > 2u || state.input_count == 0u ||
      state.input_count != run.input_count ||
      state.graph_input_resource_count != state.input_count ||
      inputs.size() != run.input_count ||
      state.input_count > VirtualPipelineState::InputCapacity) {
    return false;
  }

  const residency::TiledGraphPlan &graph =
      state.pipeline->residency->tiled_graph();
  const std::span<const residency::TiledGraphStage> stages = graph.stages();
  if (stages.size() < 2u ||
      stages.size() > graph_reduce::WavefrontStageCapacity ||
      stages.front().ports.size() < 2u || stages.back().ports.size() < 2u) {
    return false;
  }

  std::array<bool, VirtualPipelineState::InputCapacity> external_seen{};
  std::size_t external_resource_count = 0u;
  std::size_t output_resource_count = 0u;
  for (const residency::TiledGraphResource &resource : graph.resources()) {
    if (resource.resource == 0u || resource.physical_id == 0u ||
        resource.page_bytes == 0u || resource.logical_bytes == 0u) {
      return false;
    }
    if (resource.kind == residency::GraphResourceKind::ExternalInput) {
      if (resource.role != residency::GraphResourceRole::Input ||
          resource.persistence != residency::ResourcePersistence::Backing ||
          resource.producer_stage != residency::NoGraphStage) {
        return false;
      }
      ++external_resource_count;
    } else if (resource.kind == residency::GraphResourceKind::ExternalOutput) {
      if (resource.role != residency::GraphResourceRole::Output ||
          resource.persistence != residency::ResourcePersistence::Backing ||
          resource.producer_stage != residency::NoGraphStage) {
        return false;
      }
      ++output_resource_count;
    } else if (resource.kind == residency::GraphResourceKind::Internal) {
      if (resource.role != residency::GraphResourceRole::Intermediate ||
          resource.persistence != residency::ResourcePersistence::Transient ||
          resource.producer_stage >= stages.size()) {
        return false;
      }
    } else {
      return false;
    }
  }
  if (external_resource_count != state.input_count ||
      output_resource_count != 1u) {
    return false;
  }

  for (std::size_t input_index = 0u; input_index < state.input_count;
       ++input_index) {
    VirtualBacking *const input = inputs[input_index];
    const std::uint32_t resource_id = state.graph_input_resources[input_index];
    const residency::TiledGraphResource *const resource =
        graph.resource(resource_id);
    if (input == nullptr || input->max_parallel_reads() > 1u ||
        VirtualBackingAccess::resident(*input) != nullptr ||
        resource == nullptr ||
        resource->kind != residency::GraphResourceKind::ExternalInput ||
        resource->role != residency::GraphResourceRole::Input ||
        resource->persistence != residency::ResourcePersistence::Backing) {
      return false;
    }
  }

  for (std::size_t stage_index = 0u; stage_index < stages.size();
       ++stage_index) {
    const residency::TiledGraphStage &stage = stages[stage_index];
    if (stage.ports.size() < 2u ||
        stage.ports.size() > residency::TiledGraphPortCapacity) {
      return false;
    }
    std::size_t read_count = 0u;
    std::size_t external_read_count = 0u;
    std::size_t write_count = 0u;
    for (const residency::TiledGraphPort port : stage.ports) {
      const residency::TiledGraphResource *const resource =
          graph.resource(port.resource);
      if (resource == nullptr || (port.access != residency::Access::Read &&
                                  port.access != residency::Access::Write)) {
        return false;
      }
      if (port.access == residency::Access::Write) {
        if (++write_count != 1u) {
          return false;
        }
        const bool terminal = stage_index + 1u == stages.size();
        if (terminal) {
          if (resource->kind != residency::GraphResourceKind::ExternalOutput ||
              resource->role != residency::GraphResourceRole::Output ||
              resource->persistence !=
                  residency::ResourcePersistence::Backing) {
            return false;
          }
        } else if (resource->kind != residency::GraphResourceKind::Internal ||
                   resource->role !=
                       residency::GraphResourceRole::Intermediate ||
                   resource->persistence !=
                       residency::ResourcePersistence::Transient ||
                   resource->producer_stage != stage_index) {
          return false;
        }
        continue;
      }

      ++read_count;
      if (resource->kind == residency::GraphResourceKind::ExternalInput) {
        if (++external_read_count != 1u ||
            resource->persistence != residency::ResourcePersistence::Backing ||
            resource->role != residency::GraphResourceRole::Input ||
            resource->producer_stage != residency::NoGraphStage) {
          return false;
        }
        const auto found = std::find(
            state.graph_input_resources.begin(),
            state.graph_input_resources.begin() +
                static_cast<std::ptrdiff_t>(state.graph_input_resource_count),
            resource->resource);
        if (found ==
            state.graph_input_resources.begin() +
                static_cast<std::ptrdiff_t>(state.graph_input_resource_count)) {
          return false;
        }
        const std::size_t input_index = static_cast<std::size_t>(
            std::distance(state.graph_input_resources.begin(), found));
        if (input_index >= external_seen.size()) {
          return false;
        }
        external_seen[input_index] = true;
      } else if (resource->kind == residency::GraphResourceKind::Internal) {
        if (resource->role != residency::GraphResourceRole::Intermediate ||
            resource->persistence !=
                residency::ResourcePersistence::Transient ||
            resource->producer_stage == residency::NoGraphStage ||
            resource->producer_stage >= stage_index) {
          return false;
        }
      } else {
        return false;
      }
    }
    if (read_count == 0u || write_count != 1u) {
      return false;
    }
  }
  return std::all_of(external_seen.begin(),
                     external_seen.begin() +
                         static_cast<std::ptrdiff_t>(state.input_count),
                     [](const bool seen) { return seen; });
}

namespace {

[[nodiscard]] bool graph_input_span_matches(
    const VirtualPipelineState &state, const VirtualRunProjection &run,
    const std::span<VirtualBacking *const> inputs) noexcept {
  if (inputs.size() != run.input_count) {
    return false;
  }
  for (std::size_t index = 0u; index < run.input_count; ++index) {
    const VirtualBufferState *const buffer = virtual_input(state, index);
    if (buffer == nullptr || buffer->backing == nullptr ||
        inputs[index] != buffer->backing.get()) {
      return false;
    }
  }
  return true;
}

} // namespace

GraphResidentDecision
graph_resident_decision(const VirtualPipelineState &state,
                        const std::span<VirtualBacking *const> inputs,
                        const VirtualRunProjection &run) noexcept {
  if (!run.graph_execution() || run.graph_reduction() || run.reduction() ||
      state.geometry.route != VirtualRoute::GraphPointwise ||
      state.pipeline == nullptr || state.pipeline->residency == nullptr ||
      !state.pipeline->residency->graph_tiled() ||
      !device_vsm_product_detail::graph_resident_geometry(
          state.pipeline->residency->tiled_graph(), run)) {
    return {};
  }

  device_vsm_product_detail::EndpointSet endpoints{};
  const device_vsm_product_detail::EndpointMode endpoint =
      device_vsm_product_detail::classify_endpoints(state, run, endpoints);
  if (endpoint == device_vsm_product_detail::EndpointMode::Invalid) {
    return {GraphResidentDecisionKind::Decline,
            device_vsm_product_detail::EndpointMode::Invalid};
  }
  if (endpoint == device_vsm_product_detail::EndpointMode::Resident) {
    return {GraphResidentDecisionKind::Accepted, endpoint};
  }
  if (!graph_input_span_matches(state, run, inputs)) {
    return {GraphResidentDecisionKind::Decline,
            device_vsm_product_detail::EndpointMode::Invalid};
  }
  if (run.input_count == 1u) {
    return {GraphResidentDecisionKind::Accepted, endpoint};
  }
  return device_vsm_product_detail::graph_read_cohort(inputs) == nullptr
             ? GraphResidentDecision{}
             : GraphResidentDecision{GraphResidentDecisionKind::Accepted,
                                     endpoint};
}

} // namespace rund::compute::detail::device_vsm_route_detail
