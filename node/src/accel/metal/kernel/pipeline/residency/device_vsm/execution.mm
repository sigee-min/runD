#include "internal.hpp"

#include "../../../../../clock.hpp"
#include "../../../../command/submit.hpp"

#include <algorithm>
#include <cstring>

namespace rund::node::accel::detail::metal_device_vsm {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

namespace {

void accept_submission(Owner &owner) noexcept {
  if (owner.pending_request.submission_control != nullptr) {
    (void)owner.pending_request.submission_control->accept();
  }
}

} // namespace

DeviceVsmNativeExecution execute(Owner &owner, KernelCompletion completion,
                                 void *const user) noexcept {
  if (owner.adapter == nullptr || owner.adapter->queue == nullptr ||
      owner.pipeline == nullptr || owner.proof == nullptr) {
    return {.accepted = {false, "accel_metal_command_unavailable"}};
  }
  id<MTLCommandQueue> queue =
      (__bridge id<MTLCommandQueue>)owner.adapter->queue.get();
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)owner.pipeline.get();
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  if (queue == nil || pipeline == nil || command == nil || encoder == nil) {
    return {.accepted = {false, "accel_metal_command_unavailable"}};
  }
  const DeviceVsmProof &proof = *owner.proof;
  const bool graph = proof.topology == DeviceVsmTopology::GraphResident;
  const bool graph_map =
      graph && device_vsm_page_map_active(proof.graph_resident.page_map);
  const bool graph_pointwise_map =
      proof.topology == DeviceVsmTopology::GraphPointwise &&
      device_vsm_page_map_active(proof.graph_pointwise.page_map);
  if (graph_map || graph_pointwise_map) {
    const DeviceVsmPageMap &page_map = graph_map
                                           ? proof.graph_resident.page_map
                                           : proof.graph_pointwise.page_map;
    id<MTLBuffer> map = (__bridge id<MTLBuffer>)owner.params.get();
    if (map == nil || [map contents] == nullptr ||
        static_cast<std::uint64_t>([map length]) != DeviceVsmPageMapBytes ||
        static_cast<std::uint64_t>([map allocatedSize]) <
            DeviceVsmPageMapBytes) {
      return {.accepted = {false, "accel_metal_buffer_unavailable"}};
    }
    const auto *const words =
        static_cast<const std::uint32_t *>([map contents]);
    for (std::size_t index = 0u; index < page_map.words.size(); ++index) {
      if (words[index] != page_map.words[index]) {
        return {.accepted = {false, "accel_metal_buffer_unavailable"}};
      }
    }
  }
  const bool ring = proof.topology == DeviceVsmTopology::Pointwise ||
                    proof.topology == DeviceVsmTopology::GraphPointwise;
  const bool window_ring = proof.topology == DeviceVsmTopology::Window &&
                           proof.window.ring.gpu_owned;
  DeviceVsmWindowRingPlan window_ring_plan{};
  DeviceVsmWindowRingResult window_ring_result{};
  if (window_ring &&
      (!device_vsm_window_ring_plan_expected(proof.geometry,
                                             window_ring_plan) ||
       proof.window.ring != window_ring_plan ||
       !device_vsm_window_ring_result_expected(proof.geometry, window_ring_plan,
                                               window_ring_result) ||
       owner.config == nullptr || owner.ring_state == nullptr ||
       owner.ring_scratch_count != DeviceVsmWindowRingSlotCount)) {
    return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
  }
  id<MTLBuffer> window_config =
      window_ring ? (__bridge id<MTLBuffer>)owner.config.get() : nil;
  id<MTLBuffer> window_state =
      window_ring ? (__bridge id<MTLBuffer>)owner.ring_state.get() : nil;
  if (window_ring &&
      (window_config == nil || [window_config contents] == nullptr ||
       static_cast<std::uint64_t>([window_config allocatedSize]) <
           window_ring_plan.config_bytes ||
       window_state == nil || [window_state contents] == nullptr ||
       static_cast<std::uint64_t>([window_state allocatedSize]) <
           window_ring_plan.state_bytes)) {
    return {.accepted = {false, "accel_metal_buffer_unavailable"}};
  }
  if (window_ring) {
    for (std::size_t slot = 0u; slot < DeviceVsmWindowRingSlotCount; ++slot) {
      id<MTLBuffer> scratch =
          (__bridge id<MTLBuffer>)owner.ring_scratch[slot].get();
      if (scratch == nil || [scratch contents] == nullptr ||
          static_cast<std::uint64_t>([scratch allocatedSize]) <
              window_ring_plan.scratch_bytes / DeviceVsmWindowRingSlotCount) {
        return {.accepted = {false, "accel_metal_buffer_unavailable"}};
      }
    }
  }
  if (ring && owner.ring_scratch_count != owner.resident_count) {
    return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
  }
  if (window_ring && owner.ring_scratch_count != DeviceVsmWindowRingSlotCount) {
    return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
  }
  if (graph) {
    if (owner.graph.ready == 0u || owner.graph.table == nullptr ||
        owner.graph.owner_count == 0u || owner.graph.endpoint_count == 0u ||
        owner.graph_icb == nullptr || owner.graph_pipeline != owner.pipeline) {
      return {.accepted = {false, "accel_kernel_pipeline_invalid"}};
    }
    [encoder useResource:(__bridge id<MTLBuffer>)owner.params.get()
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:(__bridge id<MTLBuffer>)owner.graph.table.get()
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:(__bridge id<MTLBuffer>)owner.result.get()
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    for (std::size_t slot = 0u; slot < owner.graph.owner_count; ++slot) {
      for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
           ++bank) {
        [encoder
            useResource:(__bridge id<MTLBuffer>)owner.graph.owners[slot][bank]
                            .device_buffer.get()
                  usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      }
    }
    for (std::size_t endpoint = 0u; endpoint < owner.graph.endpoint_count;
         ++endpoint) {
      [encoder
          useResource:(__bridge id<MTLBuffer>)owner.graph.endpoints[endpoint]
                          .device_buffer.get()
                usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
  } else {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(__bridge id<MTLBuffer>)owner.params.get()
                offset:0u
               atIndex:0u];
    for (std::size_t index = 0u; index < owner.resident_count; ++index) {
      const DeviceVsmResidentBinding &row = proof.residents.rows[index];
      [encoder
          setBuffer:ring ? (__bridge id<MTLBuffer>)owner.ring_scratch[index]
                               .get()
                         : (__bridge id<MTLBuffer>)owner.residents[index]
                               .device_buffer.get()
             offset:ring ? 0u
                         : static_cast<NSUInteger>(row.backing.offset_bytes)
            atIndex:static_cast<NSUInteger>(index + 1u)];
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)owner.config.get()
                offset:0u
               atIndex:static_cast<NSUInteger>(owner.resident_count + 1u)];
    [encoder setBuffer:(__bridge id<MTLBuffer>)owner.result.get()
                offset:0u
               atIndex:static_cast<NSUInteger>(owner.resident_count + 2u)];
  }
  if (ring) {
    const std::size_t inputs = proof.residents.input_count;
    const std::size_t output = proof.residents.input_count;
    [encoder setBuffer:(__bridge id<MTLBuffer>)owner.ring_state.get()
                offset:0u
               atIndex:static_cast<NSUInteger>(owner.resident_count + 3u)];
    [encoder setBuffer:(__bridge id<MTLBuffer>)owner.ring_scratch[output].get()
                offset:0u
               atIndex:static_cast<NSUInteger>(owner.resident_count + 4u)];
    for (std::size_t input = 0u; input < inputs; ++input) {
      [encoder
          setBuffer:(__bridge id<MTLBuffer>)owner.ring_scratch[input].get()
             offset:0u
            atIndex:static_cast<NSUInteger>(owner.resident_count + 5u + input)];
      [encoder setBuffer:(__bridge id<MTLBuffer>)owner.residents[input]
                             .device_buffer.get()
                  offset:static_cast<NSUInteger>(
                             proof.residents.rows[input].backing.offset_bytes)
                 atIndex:static_cast<NSUInteger>(owner.resident_count + 5u +
                                                 inputs + input)];
    }
    [encoder setBuffer:(__bridge id<MTLBuffer>)owner.residents[output]
                           .device_buffer.get()
                offset:static_cast<NSUInteger>(
                           proof.residents.rows[output].backing.offset_bytes)
               atIndex:static_cast<NSUInteger>(owner.resident_count + 5u +
                                               2u * inputs)];
  }
  if (window_ring) {
    [encoder setBuffer:window_state offset:0u atIndex:5u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)owner.ring_scratch[0u].get()
                offset:0u
               atIndex:6u];
    [encoder setBuffer:(__bridge id<MTLBuffer>)owner.ring_scratch[1u].get()
                offset:0u
               atIndex:7u];
  }
  const NSUInteger threads =
      proof.topology == DeviceVsmTopology::Window ? proof.window.workgroup_width
      : proof.topology == DeviceVsmTopology::GraphPointwise
          ? proof.graph_pointwise.workgroup_width
      : proof.topology == DeviceVsmTopology::GraphMapReduce
          ? proof.graph_map_reduce.workgroup_width
      : proof.topology == DeviceVsmTopology::Scan ? proof.scan.workgroup_width
      : proof.topology == DeviceVsmTopology::Reduce
          ? proof.reduce.workgroup_width
          : std::max<NSUInteger>(
                1u, std::min<NSUInteger>(
                        256u, [pipeline maxTotalThreadsPerThreadgroup]));
  const NSUInteger groups =
      window_ring || graph ||
              proof.topology == DeviceVsmTopology::GraphPointwise ||
              proof.topology == DeviceVsmTopology::GraphMapReduce ||
              proof.topology == DeviceVsmTopology::Scan ||
              proof.topology == DeviceVsmTopology::Reduce
          ? 1u
          : proof.width;
  std::uint32_t window_seed_dispatches = 0u;
  std::uint32_t window_compute_dispatches = 0u;
  std::uint32_t window_internal_dispatches = 0u;
  std::uint32_t encoded_tile_count = 0u;
  if (window_ring && !device_vsm_window_internal_dispatch_count_native(
                         window_ring_plan.page_count,
                         window_internal_dispatches)) {
    return {.accepted = {false, DeviceVsmDispatchCountOverflowReason}};
  }
  if (window_ring) {
    id<MTLBuffer> config = window_config;
    [encoder setBuffer:config offset:0u atIndex:3u];
    [encoder dispatchThreadgroups:MTLSizeMake(1u, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
    window_seed_dispatches = window_ring_plan.page_count;
    window_compute_dispatches = window_ring_plan.page_count;
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  } else if (graph) {
    const MetalIcbChunk chunk{
        .commands =
            (__bridge id<MTLIndirectCommandBuffer>)owner.graph_icb.get(),
        .command_count = 1u,
        .flags = 0u,
    };
    if (!EncodeMetalPipelineIcbChunks(encoder, &chunk, 1u).ok) {
      return {.accepted = {false, "accel_metal_command_unavailable"}};
    }
    encoded_tile_count = DeviceVsmGraphPhysicalDispatchCount;
  } else {
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
  }
  [encoder endEncoding];

  owner.pending_native = DeviceVsmNativeExecution{
      .accepted = {true, "ok"},
      .window_seed_dispatches = window_seed_dispatches,
      .window_compute_dispatches = window_compute_dispatches,
      .window_internal_dispatches = window_internal_dispatches,
      .encoded_tile_count = encoded_tile_count,
  };
  owner.submit_begin_ns = MonotonicNanoseconds();
  const rund::AccelCheck queued = QueueCommand(
      *owner.adapter, (__bridge void *)command, completion, user, true);
  if (!queued.ok) {
    return {.accepted = queued};
  }
  accept_submission(owner);
  return owner.pending_native;
}

#endif

} // namespace rund::node::accel::detail::metal_device_vsm
