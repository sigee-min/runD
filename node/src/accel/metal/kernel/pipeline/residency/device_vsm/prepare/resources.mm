#include "internal.hpp"

#include "../../../../../adapter.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace rund::node::accel::detail::metal_device_vsm::prepare {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

namespace {

[[nodiscard]] std::shared_ptr<void> buffer(id<MTLDevice> device,
                                           const NSUInteger bytes) noexcept {
  if (device == nil || bytes == 0u) {
    return {};
  }
  id<MTLBuffer> value =
      [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  return value == nil ? std::shared_ptr<void>{}
                      : RetainMetalObject((__bridge void *)value);
}

} // namespace

rund::AccelCheck AllocateResources(Owner &owner, MetalAdapter &adapter,
                                   const DeviceVsmProof &proof,
                                   const Shape &shape,
                                   ResourcePlan &plan) noexcept {
  plan = {};
  rund::kernel::LoweringArtifact artifact = *proof.artifact;
  owner.pipeline = MetalPipelineForArtifact(adapter, std::move(artifact));
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  const bool map_storage = shape.graph_map || shape.graph_pointwise_map;
  const std::uint64_t raw_parameter_bytes =
      map_storage ? DeviceVsmPageMapBytes
                  : std::max<std::uint64_t>(proof.parameter_bytes, 1u);
  if (raw_parameter_bytes > std::numeric_limits<NSUInteger>::max()) {
    return {false, "compute_pipeline_capacity"};
  }
  plan.parameter_bytes = raw_parameter_bytes;
  const NSUInteger parameter_bytes =
      static_cast<NSUInteger>(plan.parameter_bytes);
  owner.params = buffer(device, parameter_bytes);
  owner.config =
      shape.graph
          ? std::shared_ptr<void>{}
          : buffer(device, shape.window_ring
                               ? static_cast<NSUInteger>(
                                     shape.window_ring_plan.config_bytes)
                               : sizeof(Config));
  if (shape.graph) {
    GraphResidentTable table{};
    table.logical_elements =
        proof.geometry.logical_bytes / proof.geometry.element_bytes;
    table.payload_elements = static_cast<std::uint32_t>(
        proof.geometry.payload_bytes / proof.geometry.element_bytes);
    table.page_count = static_cast<std::uint32_t>(proof.geometry.page_count);
    table.element_words = static_cast<std::uint32_t>(
        proof.graph_resident.type.element_bytes / sizeof(std::uint32_t));
    table.element_bytes = proof.graph_resident.type.element_bytes;
    table.valid = 1u;
    table.owner_count = owner.graph.owner_count;
    table.controller = device_vsm_graph_controller(proof.graph_wavefront);
    for (std::size_t slot = 0u; slot < owner.graph.owner_count; ++slot) {
      table.owner_valid[slot] = 1u;
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        table.owner_generations[slot * DeviceVsmGraphResidentBankCapacity +
                                bank] =
            proof.graph_resident.owners[slot].buffer_generations[bank];
      }
    }
    owner.graph.table = buffer(device, sizeof(table));
    if (owner.graph.table == nullptr) {
      return {false, "device_vsm_graph_table_failed"};
    }
    id<MTLBuffer> graph_table = (__bridge id<MTLBuffer>)owner.graph.table.get();
    if (graph_table == nil || [graph_table contents] == nullptr) {
      return {false, "device_vsm_graph_table_failed"};
    }
    std::memcpy([graph_table contents], &table, sizeof(table));
    owner.graph.ready = 1u;
  }
  owner.result =
      buffer(device, DeviceVsmResultWordCount * sizeof(std::uint32_t));
  std::uint64_t ring_state_bytes = 0u;
  std::uint64_t ring_scratch_bytes = 0u;
  std::uint64_t ring_region_bytes = 0u;
  std::uint64_t expected_ring_scratch_bytes = 0u;
  if (shape.ring &&
      (!device_vsm_ring_storage_expected(
           proof.geometry, proof.width, proof.residents.count, ring_state_bytes,
           ring_scratch_bytes) ||
       !::rund::kernel::checked::mul(proof.width, proof.geometry.payload_bytes,
                                     ring_region_bytes) ||
       !::rund::kernel::checked::mul(ring_region_bytes, proof.residents.count,
                                     expected_ring_scratch_bytes) ||
       ring_scratch_bytes != expected_ring_scratch_bytes ||
       ring_state_bytes > std::numeric_limits<NSUInteger>::max() ||
       ring_region_bytes > std::numeric_limits<NSUInteger>::max())) {
    return {false, "compute_pipeline_capacity"};
  }
  if (shape.window_ring) {
    if (shape.window_ring_plan.scratch_bytes % DeviceVsmWindowRingSlotCount !=
            0u ||
        shape.window_ring_plan.state_bytes >
            std::numeric_limits<NSUInteger>::max() ||
        shape.window_ring_plan.scratch_bytes / DeviceVsmWindowRingSlotCount >
            std::numeric_limits<NSUInteger>::max()) {
      return {false, "compute_pipeline_capacity"};
    }
    ring_state_bytes = shape.window_ring_plan.state_bytes;
    ring_region_bytes =
        shape.window_ring_plan.scratch_bytes / DeviceVsmWindowRingSlotCount;
  }
  if (shape.ring_storage) {
    owner.ring_state =
        buffer(device, static_cast<NSUInteger>(ring_state_bytes));
    owner.ring_scratch_count = shape.window_ring ? DeviceVsmWindowRingSlotCount
                                                 : proof.residents.count;
    for (std::size_t index = 0u; index < owner.ring_scratch_count; ++index) {
      owner.ring_scratch[index] =
          buffer(device, static_cast<NSUInteger>(ring_region_bytes));
    }
  }
  plan.ring_state_bytes = ring_state_bytes;
  plan.ring_region_bytes = ring_region_bytes;
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_device_vsm::prepare
