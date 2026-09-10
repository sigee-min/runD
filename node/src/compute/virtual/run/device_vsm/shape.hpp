#pragma once

#include "endpoint_mode.hpp"
#include "model.hpp"

#include "../../../type.hpp"

#include <kernel/core/checked.hpp>

#include <kernel/program/compute/window/model.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::compute::detail::device_vsm_product_detail {

[[nodiscard]] inline bool
staged_loop_shape(const VirtualPipelineState &state,
                  const VirtualRunProjection &run) noexcept {
  return !run.device_vsm_required && !run.graph_execution() &&
         !run.graph_reduction() && !run.reduction() && !run.scan() &&
         !run.multi_pointwise() && !run.multi_scan() &&
         state.geometry.route == VirtualRoute::Pointwise &&
         run.input_count == 1u && state.input_count == 1u &&
         state.inputs[0u] != nullptr && state.output != nullptr &&
         state.inputs[0u]->backing != nullptr &&
         state.output->backing != nullptr &&
         VirtualBackingAccess::resident(*state.inputs[0u]->backing) ==
             nullptr &&
         VirtualBackingAccess::resident(*state.output->backing) == nullptr &&
         run.input_page_bytes != 0u &&
         run.input_page_bytes == run.input_payload_bytes &&
         run.output_page_bytes != 0u &&
         run.output_page_bytes == run.output_payload_bytes &&
         run.input_prefix_bytes == 0u && run.output_prefix_bytes == 0u &&
         run.active.stream.page_count() >= 2u &&
         run.active.stream.epoch_count() >= 2u;
}

[[nodiscard]] inline bool
window_ring_shape(const VirtualPipelineState &state,
                  const VirtualRunProjection &run,
                  VirtualDeviceVsmEndpoint &endpoint) noexcept {
  endpoint = VirtualDeviceVsmEndpoint::Invalid;
  const VirtualWindowPreflight &preflight = state.window_preflight;
  if (preflight.mode != VirtualWindowPreflightMode::WindowRing ||
      preflight.page_count != run.active.stream.page_count() ||
      preflight.frame_capacity != run.frame_capacity ||
      preflight.endpoint == VirtualWindowPreflightEndpoint::Invalid ||
      run.input_count != 1u || state.input_count != 1u ||
      state.geometry.route != VirtualRoute::Window) {
    return false;
  }
  if (preflight.endpoint == VirtualWindowPreflightEndpoint::Resident) {
    endpoint = VirtualDeviceVsmEndpoint::Resident;
  } else if (preflight.endpoint == VirtualWindowPreflightEndpoint::Staged) {
    endpoint = VirtualDeviceVsmEndpoint::Staged;
  }
  return endpoint != VirtualDeviceVsmEndpoint::Invalid;
}

[[nodiscard]] inline bool
window_ring_shape(const VirtualPipelineState &state,
                  const VirtualRunProjection &run) noexcept {
  VirtualDeviceVsmEndpoint endpoint{};
  return window_ring_shape(state, run, endpoint);
}

[[nodiscard]] inline bool
graph_linear_chain(const residency::TiledGraphPlan &graph,
                   const std::span<const residency::TiledGraphStage> stages,
                   const std::size_t input_count) noexcept {
  if (stages.size() < 2u || input_count < 2u) {
    return false;
  }
  for (std::size_t index = 0u; index < stages.size(); ++index) {
    const residency::TiledGraphStage &stage = stages[index];
    const bool first = index == 0u;
    const bool last = index + 1u == stages.size();
    const std::size_t expected_reads = first ? input_count : 1u;
    const std::size_t expected_writes = 1u;
    std::size_t reads = 0u;
    std::size_t writes = 0u;
    const residency::TiledGraphPort *read = nullptr;
    const residency::TiledGraphPort *write = nullptr;
    for (const residency::TiledGraphPort &port : stage.ports) {
      const residency::TiledGraphResource *const resource =
          graph.resource(port.resource);
      if (resource == nullptr) {
        return false;
      }
      if (port.access == residency::Access::Read) {
        ++reads;
        read = &port;
        if (first) {
          if (resource->kind != residency::GraphResourceKind::ExternalInput ||
              resource->role != residency::GraphResourceRole::Input ||
              resource->persistence != residency::ResourcePersistence::Backing ||
              resource->first_stage != 0u || resource->last_stage != 0u ||
              resource->producer_stage != residency::NoGraphStage ||
              port.next_stage != residency::NoGraphStage) {
            return false;
          }
        } else if (resource->kind != residency::GraphResourceKind::Internal ||
                   resource->role != residency::GraphResourceRole::Intermediate ||
                   resource->persistence != residency::ResourcePersistence::Transient ||
                   resource->producer_stage != index - 1u ||
                   resource->first_stage != index - 1u ||
                   resource->last_stage != index ||
                   port.next_stage != residency::NoGraphStage) {
          return false;
        }
      } else if (port.access == residency::Access::Write) {
        ++writes;
        write = &port;
        if (last) {
          if (resource->kind != residency::GraphResourceKind::ExternalOutput ||
              resource->role != residency::GraphResourceRole::Output ||
              resource->persistence != residency::ResourcePersistence::Backing ||
              resource->first_stage != index || resource->last_stage != index ||
              resource->producer_stage != residency::NoGraphStage ||
              port.next_stage != residency::NoGraphStage) {
            return false;
          }
        } else if (resource->kind != residency::GraphResourceKind::Internal ||
                   resource->role != residency::GraphResourceRole::Intermediate ||
                   resource->persistence != residency::ResourcePersistence::Transient ||
                   resource->producer_stage != index ||
                   resource->first_stage != index ||
                   resource->last_stage != index + 1u ||
                   port.next_stage != index + 1u) {
          return false;
        }
      } else {
        return false;
      }
    }
    if (reads != expected_reads || writes != expected_writes ||
        read == nullptr || write == nullptr) {
      return false;
    }
    if (first && write->next_stage != 1u) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
graph_resident_geometry(const residency::TiledGraphPlan &graph,
                        const VirtualRunProjection &run) noexcept {
  constexpr std::uint64_t max_u32 =
      std::numeric_limits<std::uint32_t>::max();
  constexpr std::size_t stage_capacity =
      node::accel::detail::DeviceVsmGraphResidentStageCapacity;
  constexpr std::size_t resource_capacity =
      node::accel::detail::DeviceVsmGraphResidentResourceCapacity;
  constexpr std::size_t port_capacity =
      node::accel::detail::DeviceVsmGraphResidentPortCapacity;
  constexpr std::size_t owner_capacity =
      node::accel::detail::DeviceVsmGraphResidentPhysicalCapacity;
  const auto stages = graph.stages();
  const auto resources = graph.resources();
  const auto owners = graph.physical_classes();
  const Type root_type = resources.empty() ? Type::I32 : resources.front().type;
  const std::size_t element_bytes = type_bytes(root_type);
  const bool unsigned_type = root_type == Type::U32 || root_type == Type::U64;
  const std::uint64_t pages = run.active.graph.page_count();
  const std::uint64_t frames = run.frame_capacity;
  std::uint64_t batches = 0u;
  std::uint64_t steps = 0u;
  if (run.input_count < 2u || stages.size() < 2u ||
      stages.size() > stage_capacity ||
      resources.size() < 2u || resources.size() > resource_capacity ||
      owners.empty() || owners.size() > owner_capacity || pages == 0u ||
      pages > max_u32 || frames == 0u || frames > pages ||
      frames > max_u32 || !kernel::checked::add(pages, frames - 1u, batches)) {
    return false;
  }
  batches /= frames;
  if (batches == 0u || batches > max_u32 ||
      !kernel::checked::mul(static_cast<std::uint64_t>(stages.size()), batches,
                            steps) ||
      steps >= node::accel::detail::DeviceVsmGraphTileCapacity ||
      graph.page_count() != pages || graph.frame_capacity() != frames ||
      graph.batch_count() != batches || !run.active.graph.valid() ||
      run.active.graph.frame_capacity() != frames ||
      run.active.graph.batch_count() != batches ||
      run.active.input_bytes == 0u ||
      run.active.input_bytes != run.active.output_bytes ||
      run.input_page_bytes == 0u ||
      run.input_page_bytes != run.output_page_bytes ||
      run.input_payload_bytes == 0u ||
      run.input_payload_bytes != run.output_payload_bytes ||
      run.input_prefix_bytes != run.output_prefix_bytes ||
      run.input_prefix_bytes > run.input_page_bytes ||
      run.input_payload_bytes >
          run.input_page_bytes - run.input_prefix_bytes ||
      run.input_prefix_bytes !=
          run.input_page_bytes - run.input_prefix_bytes -
              run.input_payload_bytes ||
      element_bytes == 0u || !unsigned_type || run.input_type != root_type ||
      run.output_type != root_type ||
      run.input_page_bytes % element_bytes != 0u ||
      run.input_page_bytes / element_bytes > max_u32 ||
      !kernel::checked::mul(pages, run.input_page_bytes) ||
      !kernel::checked::mul(pages, run.output_page_bytes) ||
      !kernel::checked::mul(frames, run.input_page_bytes) ||
      !kernel::checked::mul(frames, run.output_page_bytes)) {
    return false;
  }

  std::size_t port_count = 0u;
  std::size_t input_count = 0u;
  std::size_t output_count = 0u;
  std::size_t internal_count = 0u;
  bool internal_fan_in = false;
  std::uint32_t prior_node = 0u;
  for (std::size_t index = 0u; index < stages.size(); ++index) {
    const residency::TiledGraphStage &stage = stages[index];
    if (stage.domain != residency::StageDomain::Tile ||
        stage.ports.empty() || stage.ports.size() > port_capacity ||
        (index != 0u && stage.node <= prior_node) ||
        port_count > port_capacity - stage.ports.size()) {
      return false;
    }
    std::size_t reads = 0u;
    std::size_t writes = 0u;
    std::size_t internal_reads = 0u;
    bool internal_write = false;
    for (std::size_t port = 0u; port < stage.ports.size(); ++port) {
      const residency::TiledGraphPort &current = stage.ports[port];
      const residency::TiledGraphResource *const resource =
          graph.resource(current.resource);
      if (resource == nullptr ||
          (current.access != residency::Access::Read &&
           current.access != residency::Access::Write) ||
          current.program_port >= port_capacity ||
          (current.next_stage != residency::NoGraphStage &&
           current.next_stage >= stages.size())) {
        return false;
      }
      for (std::size_t prior = 0u; prior < port; ++prior) {
        if (stage.ports[prior].resource == current.resource) {
          return false;
        }
      }
      if (current.access == residency::Access::Read) {
        ++reads;
        if (resource->kind == residency::GraphResourceKind::Internal &&
            resource->role == residency::GraphResourceRole::Intermediate &&
            resource->persistence == residency::ResourcePersistence::Transient) {
          ++internal_reads;
        }
      } else {
        ++writes;
        if (resource->kind == residency::GraphResourceKind::Internal &&
            resource->role == residency::GraphResourceRole::Intermediate &&
            resource->persistence == residency::ResourcePersistence::Transient) {
          internal_write = true;
        }
      }
    }
    if (reads == 0u || writes != 1u || stage.active_count_input != reads) {
      return false;
    }
    if (index + 1u < stages.size() && internal_reads >= 2u && internal_write) {
      internal_fan_in = true;
    }
    port_count += stage.ports.size();
    prior_node = stage.node;
  }
  if (port_count == 0u || port_count > port_capacity) {
    return false;
  }
  const bool linear_chain =
      graph_linear_chain(graph, stages, run.input_count);
  if (!internal_fan_in && !linear_chain) {
    return false;
  }

  for (std::size_t index = 0u; index < resources.size(); ++index) {
    const residency::TiledGraphResource &resource = resources[index];
    std::uint64_t materialized = 0u;
    if (resource.resource == 0u || resource.physical_id == 0u ||
        resource.page_bytes == 0u || resource.logical_bytes == 0u ||
        resource.type != root_type ||
        resource.page_bytes % element_bytes != 0u ||
        resource.page_bytes / element_bytes > max_u32 ||
        resource.logical_bytes % element_bytes != 0u ||
        resource.first_stage >= stages.size() ||
        resource.last_stage >= stages.size() ||
        resource.first_stage > resource.last_stage ||
        (resource.producer_stage != residency::NoGraphStage &&
         resource.producer_stage >= stages.size()) ||
        !kernel::checked::mul(pages, resource.page_bytes, materialized) ||
        resource.logical_bytes > materialized) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (resources[prior].resource == resource.resource) {
        return false;
      }
    }
    if (resource.kind == residency::GraphResourceKind::ExternalInput) {
      if (resource.role != residency::GraphResourceRole::Input ||
          resource.persistence != residency::ResourcePersistence::Backing ||
          resource.producer_stage != residency::NoGraphStage) {
        return false;
      }
      ++input_count;
    } else if (resource.kind == residency::GraphResourceKind::ExternalOutput) {
      if (resource.role != residency::GraphResourceRole::Output ||
          resource.persistence != residency::ResourcePersistence::Backing ||
          resource.producer_stage != residency::NoGraphStage) {
        return false;
      }
      ++output_count;
    } else if (resource.kind == residency::GraphResourceKind::Internal) {
      if (resource.role != residency::GraphResourceRole::Intermediate ||
          resource.persistence != residency::ResourcePersistence::Transient ||
          resource.producer_stage == residency::NoGraphStage) {
        return false;
      }
      ++internal_count;
    } else {
      return false;
    }
  }
  if (input_count != run.input_count || output_count != 1u ||
      internal_count == 0u || run.input_count == 0u ||
      run.input_count >= node::accel::detail::DeviceVsmResidentCapacity) {
    return false;
  }
  for (std::size_t index = 0u; index < owners.size(); ++index) {
    if (owners[index].physical_id == 0u) {
      return false;
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (owners[prior].physical_id == owners[index].physical_id) {
        return false;
      }
    }
  }
  return true;
}

} // namespace rund::compute::detail::device_vsm_product_detail
