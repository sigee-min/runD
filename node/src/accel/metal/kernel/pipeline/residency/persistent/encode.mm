#include "internal.hpp"

#include <algorithm>
#include <limits>

namespace rund::node::accel::detail::metal_persistent_sliding {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
encode_range(id<MTLComputeCommandEncoder> const encoder,
             const MetalWarmSubmission warm,
             const MetalResidencyStepRange range) noexcept {
  if (range.count == 0u) {
    return {true, "ok"};
  }
  if (encoder == nil || warm.chunks == nullptr || warm.chunk_count == 0u ||
      range.begin > std::numeric_limits<std::uint64_t>::max() - range.count) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  const std::uint64_t range_end = range.begin + range.count;
  std::uint64_t global = 0u;
  std::uint64_t consumed = 0u;
  for (NSUInteger index = 0u; index < warm.chunk_count; ++index) {
    const MetalIcbChunk &chunk = warm.chunks[index];
    if (!chunk.valid() || global > std::numeric_limits<std::uint64_t>::max() -
                                       chunk.command_count) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
    const std::uint64_t chunk_end = global + chunk.command_count;
    const std::uint64_t first = std::max(global, range.begin);
    const std::uint64_t last = std::min(chunk_end, range_end);
    if (last > first) {
      if (consumed != 0u || (first == global && chunk.barrier_before())) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      [encoder
          executeCommandsInBuffer:chunk.commands
                        withRange:NSMakeRange(
                                      static_cast<NSUInteger>(first - global),
                                      static_cast<NSUInteger>(last - first))];
      consumed += last - first;
    }
    global = chunk_end;
    if (global >= range_end) {
      break;
    }
  }
  return consumed == range.count
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
}

} // namespace

rund::AccelCheck
encode_selection(id<MTLComputeCommandEncoder> const encoder,
                 MetalSequence &sequence,
                 const std::span<const std::uint32_t> locals) noexcept {
  if (encoder == nil || locals.empty() || !sequence.residency_selectable ||
      locals.size() > sequence.residency_steps.size() ||
      sequence.direct_aggregate || sequence.state_count != 0u ||
      sequence.states != nil || sequence.recurrence != nullptr ||
      (sequence.spatial_window.window_present &&
       !sequence.persistent_spatial_window_selectable) ||
      (sequence.persistent_spatial_window_selectable &&
       !sequence.spatial_window_proof_valid())) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  rund::AccelCheck encoded =
      encode_range(encoder, sequence.warm, sequence.residency_prefix);
  for (std::size_t index = 0u; encoded.ok && index < locals.size(); ++index) {
    const std::uint32_t local = locals[index];
    if (local >= sequence.residency_steps.size()) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
    if (sequence.persistent_spatial_window_selectable &&
        !sequence.spatial_window.local_matches(local)) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (locals[prior] == local) {
        return {false, "accel_kernel_pipeline_invalid"};
      }
    }
    if (sequence.residency_prefix.count != 0u || index != 0u) {
      [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    encoded =
        encode_range(encoder, sequence.warm, sequence.residency_steps[local]);
  }
  if (encoded.ok && sequence.residency_suffix.count != 0u) {
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    encoded = encode_range(encoder, sequence.warm, sequence.residency_suffix);
  }
  return encoded;
}

rund::AccelCheck encode_chunk(Owner &owner, const std::uint64_t first,
                              const std::uint64_t count) noexcept {
  const PersistentResidencySlidingRequest &request = owner.prepared;
  if (owner.adapter == nullptr || owner.adapter->queue == nullptr ||
      count == 0u || first >= request.coordinate_count ||
      count > request.coordinate_count - first ||
      (request.mode == PersistentResidencySlidingMode::BackendChunked &&
       count > SlotCount)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  id<MTLCommandQueue> const queue =
      (__bridge id<MTLCommandQueue>)owner.adapter->queue.get();
  owner.command = queue == nil ? nil : [queue commandBuffer];
  if (owner.command == nil) {
    return {false, "accel_metal_command_unavailable"};
  }
  const std::uint64_t base = owner.run_event_base;
  if (base > std::numeric_limits<std::uint64_t>::max() -
                 request.coordinate_count) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  for (std::uint64_t coordinate = first; coordinate < first + count;
       ++coordinate) {
    Coordinate &entry =
        owner.coordinates[static_cast<std::size_t>(coordinate % SlotCount)];
    if (!persistent_sliding_service_identity(request, coordinate,
                                             entry.identity)) {
      return {false, "accel_kernel_pipeline_invalid"};
    }
    const PersistentResidencySlidingRole &role =
        request.roles[entry.identity.slot];
    entry.sequence = static_cast<MetalSequence *>(role.prepared.get());
    entry.locals = role.locals;
    entry.local_count = coordinate + 1u == request.coordinate_count
                            ? request.tail_local_count
                            : role.local_count;
    entry.ready_value = base + entry.identity.turn + 1u;
    entry.done_value = base + coordinate + 1u;
    [owner.command encodeWaitForEvent:owner.ready[entry.identity.slot]
                                value:entry.ready_value];
    id<MTLComputeCommandEncoder> const encoder =
        [owner.command computeCommandEncoder];
    if (encoder == nil) {
      return {false, "accel_metal_command_unavailable"};
    }
    MetalSequence &sequence = *entry.sequence;
    if (sequence.direct_aggregate || sequence.state_count != 0u ||
        sequence.states != nil || sequence.recurrence != nullptr ||
        (sequence.spatial_window.window_present &&
         !sequence.persistent_spatial_window_selectable) ||
        (sequence.persistent_spatial_window_selectable &&
         !sequence.spatial_window_proof_valid())) {
      [encoder endEncoding];
      return {false, "accel_kernel_pipeline_invalid"};
    }
    if (sequence.warm.resource_count != 0u) {
      [encoder useResources:sequence.warm.resources
                      count:sequence.warm.resource_count
                      usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    }
    [encoder useResource:sequence.residency_sliding.descriptor
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:sequence.guard_zero
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:sequence.control
                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:sequence.residency_sliding.command
                   usage:MTLResourceUsageRead];
    [encoder executeCommandsInBuffer:sequence.residency_sliding.command
                           withRange:NSMakeRange(0u, 1u)];
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    const rund::AccelCheck selected = encode_selection(
        encoder, sequence,
        std::span<const std::uint32_t>{entry.locals.data(), entry.local_count});
    [encoder endEncoding];
    if (!selected.ok) {
      return selected;
    }
    [owner.command encodeSignalEvent:owner.done value:entry.done_value];
  }
  return {true, "ok"};
}

rund::AccelCheck encode(Owner &owner) noexcept {
  owner.run_event_base = owner.event_base;
  const std::uint64_t count =
      owner.prepared.mode == PersistentResidencySlidingMode::OneSubmit
          ? owner.prepared.coordinate_count
          : owner.prepared.chunk_count;
  return encode_chunk(owner, owner.prepared.first_coordinate, count);
}

#endif

} // namespace rund::node::accel::detail::metal_persistent_sliding
