#include "../build.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::EncodeTelemetry(
    const PreparedProgramStatusSlice telemetry_slice,
    const PreparedProgramStatusSlice binding_slice,
    const std::uint32_t declared_step) {
  const std::size_t telemetry_end =
      static_cast<std::size_t>(telemetry_slice.first) + telemetry_slice.count;
  const std::size_t binding_end =
      static_cast<std::size_t>(binding_slice.first) + binding_slice.count;
  if (telemetry_end > pipeline->telemetry.size() ||
      binding_end > status_bindings.size() ||
      declared_step >= status.declared_step_count) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  if (telemetry_slice.count == 0u) {
    return rund::AccelCheck{true, "ok"};
  }
  // A reused recurrence route owns mutable primitive telemetry. Consume
  // this occurrence before the next route writes the same native buffers.
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  for (std::size_t telemetry_index = telemetry_slice.first;
       telemetry_index < telemetry_end; ++telemetry_index) {
    const MetalPipelineTelemetrySource &source =
        pipeline->telemetry[telemetry_index].source;
    const auto resolve = [&](void *const native, const std::uint64_t relative,
                             const std::uint64_t bytes,
                             id<MTLBuffer> __strong &buffer,
                             NSUInteger &offset) noexcept {
      if (native == nullptr ||
          relative > std::numeric_limits<NSUInteger>::max()) {
        return false;
      }
      buffer = (__bridge id<MTLBuffer>)native;
      std::uint64_t base = 0u;
      std::uint64_t resolved_relative = relative;
      for (std::size_t ordinal = binding_slice.first; ordinal < binding_end;
           ++ordinal) {
        const MetalPipelineStatusBindingRecord &record =
            status_bindings[ordinal];
        if (!record.binding.replace || record.binding.buffer != native ||
            relative < record.binding.offset) {
          continue;
        }
        const std::uint64_t local = relative - record.binding.offset;
        if (local > record.binding.bytes ||
            bytes > record.binding.bytes - local) {
          return false;
        }
        buffer = pipeline->raw_status;
        base = static_cast<std::uint64_t>(record.raw_offset) *
               sizeof(std::uint32_t);
        resolved_relative = local;
        break;
      }
      if (buffer == nil ||
          resolved_relative >
              std::numeric_limits<std::uint64_t>::max() - base ||
          base + resolved_relative > std::numeric_limits<NSUInteger>::max()) {
        return false;
      }
      offset = static_cast<NSUInteger>(base + resolved_relative);
      return true;
    };
    const std::uint64_t primary_bytes =
        static_cast<std::uint64_t>(source.primary_word_count) *
        sizeof(std::uint32_t);
    id<MTLBuffer> primary = nil;
    NSUInteger primary_offset = 0u;
    if (!resolve(source.primary_buffer, 0u, primary_bytes, primary,
                 primary_offset)) {
      return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
    }
    const auto scalar_bytes = [](const rund::kernel::GraphControlSource type) {
      return type == rund::kernel::GraphControlSource::U64 ? 8u : 4u;
    };
    id<MTLBuffer> count = primary;
    NSUInteger count_offset = primary_offset;
    if (source.control.has_count()) {
      const std::uint64_t bytes = scalar_bytes(source.control.count_source);
      if ((source.count_buffer == nullptr &&
           (source.count_offset > primary_bytes ||
            bytes > primary_bytes - source.count_offset)) ||
          !resolve(source.count_buffer == nullptr ? source.primary_buffer
                                                  : source.count_buffer,
                   source.count_offset, bytes, count, count_offset)) {
        return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
      }
    }
    id<MTLBuffer> predicate = primary;
    NSUInteger predicate_offset = primary_offset;
    if (source.control.has_predicate()) {
      const std::uint64_t bytes = scalar_bytes(source.control.predicate_source);
      if ((source.predicate_buffer == nullptr &&
           (source.predicate_offset > primary_bytes ||
            bytes > primary_bytes - source.predicate_offset)) ||
          !resolve(source.predicate_buffer == nullptr ? source.primary_buffer
                                                      : source.predicate_buffer,
                   source.predicate_offset, bytes, predicate,
                   predicate_offset)) {
        return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
      }
    }
    const MetalPipelineTelemetryParams params{
        .kind = static_cast<std::uint32_t>(source.kind),
        .primary_word_count = source.primary_word_count,
        .count_source = static_cast<std::uint32_t>(source.control.count_source),
        .predicate_source =
            static_cast<std::uint32_t>(source.control.predicate_source),
        .has_count = source.control.has_count() ? 1u : 0u,
        .has_predicate = source.control.has_predicate() ? 1u : 0u,
        .iteration = source.control.iteration,
        .count_word_offset = 0u,
        .predicate_word_offset = 0u,
        .indirect_dispatch_count = source.indirect_dispatch_count,
        .declared_step_count = status.declared_step_count,
        .declared_step = declared_step,
        .capacity = source.capacity,
        .predicate_expected = source.control.predicate_expected,
        .work_item_count = source.work_item_count,
    };
    [encoder setComputePipelineState:telemetry];
    [encoder setBuffer:primary offset:primary_offset atIndex:0u];
    [encoder setBuffer:count offset:count_offset atIndex:1u];
    [encoder setBuffer:predicate offset:predicate_offset atIndex:2u];
    [encoder setBuffer:pipeline->control offset:0u atIndex:3u];
    [encoder setBytes:&params length:sizeof(params) atIndex:4u];
    if (profile_steps) {
      [encoder setBuffer:pipeline->step_control offset:0u atIndex:5u];
      [encoder setBytes:&declared_step length:sizeof(declared_step) atIndex:6u];
    }
    id<MTLBuffer> const states =
        pipeline->states == nil ? pipeline->control : pipeline->states;
    if (states == nil) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
    [encoder setBuffer:states offset:0u atIndex:7u];
    [encoder setBytes:&captured.owner
               length:sizeof(captured.owner)
              atIndex:8u];
    [encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    captured.commands.back().control = true;
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
