#include "internal.hpp"

#include "../../../../../kernel/recurrence.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

namespace {

[[nodiscard]] bool
SameResidentRef(const rund::kernel::ResidentBufferRef &left,
                const rund::kernel::ResidentBufferRef &right) noexcept {
  return left.id == right.id && left.bytes == right.bytes &&
         left.offset_bytes == right.offset_bytes &&
         left.element_bytes == right.element_bytes &&
         left.stride_bytes == right.stride_bytes && left.count == right.count &&
         left.usage == right.usage;
}

[[nodiscard]] bool ValidHistoryOutput(const MetalMapEncodeResources &map,
                                      const MapRecurrenceHistory &history,
                                      const std::uint64_t index) noexcept {
  const rund::kernel::ResidentBufferRef &ref = history.outputs[index];
  const std::shared_ptr<void> &owner = history.handles[index];
  const rund::kernel::ResidentBufferRef *const bound =
      map.bindings.resident_outputs.ref(index);
  const std::shared_ptr<void> *const bound_owner =
      map.bindings.resident_outputs.handle(index);
  if (bound == nullptr || bound_owner == nullptr || owner == nullptr ||
      *bound_owner != owner || !SameResidentRef(*bound, ref) || ref.id == 0u ||
      ref.bytes == 0u || ref.offset_bytes > ref.bytes || ref.count == 0u ||
      ref.element_bytes == 0u || ref.stride_bytes < ref.element_bytes ||
      ref.usage != rund::kernel::kResidentUsageWrite ||
      ref.count % map.iterations != 0u) {
    return false;
  }
  const std::uint64_t slice_count = ref.count / map.iterations;
  if (slice_count == 0u ||
      slice_count >
          std::numeric_limits<std::uint64_t>::max() / ref.stride_bytes ||
      history.pitch_bytes[index] != slice_count * ref.stride_bytes) {
    return false;
  }
  const std::uint64_t last = ref.count - 1u;
  if (last > std::numeric_limits<std::uint64_t>::max() / ref.stride_bytes ||
      ref.offset_bytes >
          std::numeric_limits<std::uint64_t>::max() - last * ref.stride_bytes ||
      ref.element_bytes > std::numeric_limits<std::uint64_t>::max() -
                              ref.offset_bytes - last * ref.stride_bytes ||
      ref.offset_bytes + last * ref.stride_bytes + ref.element_bytes >
          ref.bytes) {
    return false;
  }
  const MetalResidentBufferResult &resident = map.resident.output(index);
  return resident.check.ok && resident.device_buffer != nullptr &&
         resident.handle == owner;
}

[[nodiscard]] bool
ValidMetalHistory(const MetalMapEncodeResources &map) noexcept {
  if (map.binding_owner == nullptr || map.prepared == nullptr ||
      map.prepared->plan.output_buffer_count == 0u || map.iterations < 2u ||
      map.bindings.resident_outputs.count !=
          map.prepared->plan.output_buffer_count ||
      !map.bindings.resident_outputs.has_refs() ||
      !map.bindings.resident_outputs.has_handles() ||
      map.resident.bindings != &map.bindings) {
    return false;
  }
  const auto *const history =
      static_cast<const MapRecurrenceHistory *>(map.binding_owner.get());
  if (history == nullptr ||
      history->count != map.prepared->plan.output_buffer_count) {
    return false;
  }
  const std::uint64_t output_count = map.prepared->plan.output_buffer_count;
  if (output_count > kInlineMetalBufferCount &&
      map.resident.overflow_outputs.size() !=
          output_count - kInlineMetalBufferCount) {
    return false;
  }
  for (std::uint64_t index = 0u; index < output_count; ++index) {
    if (!ValidHistoryOutput(map, *history, index)) {
      return false;
    }
  }
  return true;
}

} // namespace

namespace metal_persistent_sliding {

void record_first_failure(
    Owner &owner, const MetalPersistentResidencySlidingDiagnosticStage stage,
    const std::uint64_t stage_key,
    const MetalPersistentResidencySlidingDiagnosticPredicate predicate,
    const std::uint64_t predicate_key,
    const std::uint64_t coordinate) noexcept {
  if (owner.first_failure_trace_recorded) {
    return;
  }
  MetalPersistentResidencySlidingDiagnosticTrace trace{};
  trace.stage = stage;
  trace.stage_key = stage_key;
  trace.predicate = predicate;
  trace.predicate_key = predicate_key;
  trace.coordinate = coordinate;
  owner.first_failure_trace = trace;
  owner.first_failure_trace_recorded = true;
}

void record_first_failure(
    Owner &owner,
    const MetalPersistentResidencySlidingDiagnosticTrace &trace) noexcept {
  if (owner.first_failure_trace_recorded) {
    return;
  }
  owner.first_failure_trace = trace;
  owner.first_failure_trace_recorded = true;
}

void record_wait_trace(
    Owner &owner,
    const MetalPersistentResidencySlidingDiagnosticTrace &trace) noexcept {
  owner.last_wait_trace = trace;
}

} // namespace metal_persistent_sliding

bool InspectMetalPersistentResidencySliding(
    const std::shared_ptr<void> &lowering,
    MetalPersistentResidencySlidingDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  const std::shared_ptr<metal_persistent_sliding::Owner> owner =
      metal_persistent_sliding::owner_of(lowering);
  if (owner == nullptr) {
    return false;
  }
  std::lock_guard lock{owner->gate};
  if (!owner->sealed.valid) {
    return false;
  }
  diagnostics = owner->sealed.value;
  return true;
}

bool InspectMetalFusedDirectRecurrence(
    const std::shared_ptr<void> &prepared,
    MetalFusedDirectRecurrenceDiagnostics &diagnostics) noexcept {
  diagnostics = {};
  const auto *const sequence =
      static_cast<const MetalSequence *>(prepared.get());
  const auto *const map = sequence == nullptr
                              ? nullptr
                              : static_cast<const MetalMapEncodeResources *>(
                                    sequence->recurrence.get());
  if (sequence != nullptr) {
    diagnostics.native_command_count = sequence->command_count;
    diagnostics.native_dispatch_count = sequence->dispatch_count;
    diagnostics.retained_bytes = sequence->retained_bytes;
  }
  if (map != nullptr) {
    diagnostics.iteration_count = map->iterations;
    diagnostics.iteration_argument_bytes = sizeof(map->iterations);
  }
  if (!ValidMetalSequence(sequence) || map == nullptr ||
      map->prepared == nullptr || map->adapter != sequence->adapter ||
      map->controlled() || map->iterations < 2u || map->windows.size() != 1u ||
      map->windows.capacity() != 1u || !sequence->transducers.empty() ||
      sequence->state_count != 0u || sequence->direct_aggregate ||
      sequence->command_count == 0u || sequence->dispatch_count != 1u ||
      sequence->command_chunks.empty() || map->param.buffer == nullptr ||
      (map->binding_owner != nullptr && !ValidMetalHistory(*map))) {
    return false;
  }
  std::uint64_t icb_bytes = 0u;
  for (const MetalIcbChunk &chunk : sequence->command_chunks) {
    id<MTLIndirectCommandBuffer> const commands = chunk.commands;
    const std::uint64_t allocated =
        commands == nil ? 0u
                        : static_cast<std::uint64_t>(commands.allocatedSize);
    if (!chunk.valid() || commands.size < chunk.command_count ||
        allocated == 0u ||
        icb_bytes > std::numeric_limits<std::uint64_t>::max() - allocated) {
      return false;
    }
    icb_bytes += allocated;
  }
  const std::uint64_t param_bytes = map->param.allocated_bytes == 0u
                                        ? map->param.bytes
                                        : map->param.allocated_bytes;
  std::uint64_t route_host_bytes = sizeof(MetalMapEncodeResources);
  const auto add_route_rows = [&](const std::size_t count,
                                  const std::uint64_t row_bytes) noexcept {
    if (count > std::numeric_limits<std::uint64_t>::max() / row_bytes) {
      return false;
    }
    const std::uint64_t bytes = static_cast<std::uint64_t>(count) * row_bytes;
    if (route_host_bytes > std::numeric_limits<std::uint64_t>::max() - bytes) {
      return false;
    }
    route_host_bytes += bytes;
    return true;
  };
  if (param_bytes == 0u ||
      icb_bytes > std::numeric_limits<std::uint64_t>::max() - param_bytes ||
      !add_route_rows(map->windows.capacity(),
                      sizeof(rund::kernel::ComputeDispatchWindow)) ||
      !add_route_rows(map->resident.overflow_inputs.capacity(),
                      sizeof(MetalResidentBufferResult)) ||
      !add_route_rows(map->resident.overflow_outputs.capacity(),
                      sizeof(MetalResidentBufferResult))) {
    return false;
  }
  diagnostics = MetalFusedDirectRecurrenceDiagnostics{
      .iteration_count = map->iterations,
      .native_command_count = sequence->command_count,
      .native_dispatch_count = sequence->dispatch_count,
      .native_storage_bytes = icb_bytes + param_bytes,
      .route_host_bytes = route_host_bytes,
      .retained_bytes = sequence->retained_bytes,
      .iteration_argument_bytes = sizeof(map->iterations),
      .retention = map->binding_owner == nullptr
                       ? MetalFusedDirectRecurrenceRetention::Terminal
                       : MetalFusedDirectRecurrenceRetention::History,
  };
  return diagnostics.retained_bytes != 0u;
}

#endif

} // namespace rund::node::accel::detail
