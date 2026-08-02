#include "../build.hpp"

#include "../aggregate/prepare.hpp"

#include <algorithm>
#include <new>
#include <stdexcept>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Capture() {
  if (finished) {
    return rund::AccelCheck{true, "ok"};
  }
  const PreparedKernelPipelineReservation &limit = template_registry.limit;
  const MetalCaptureRowCapacity command_rows =
      PlanMetalCaptureRowCapacity(limit.backend_command_count);
  const MetalCaptureRowCapacity binding_rows =
      PlanMetalCaptureRowCapacity(limit.backend_command_binding_count);
  const MetalCaptureRowCapacity parameter_rows =
      PlanMetalCaptureRowCapacity(limit.backend_parameter_bytes);
  if (!limit.ok || !command_rows.ok || !binding_rows.ok || !parameter_rows.ok ||
      limit.backend_command_binding_slot_upper == 0u ||
      limit.backend_command_binding_slot_upper > kMetalPipelineGuardBinding ||
      encoder != nil || !captured.commands.empty() ||
      !captured.command_bindings.empty() || !captured.parameters.empty() ||
      captured.parameters.capacity() != 0u) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  try {
    captured.commands.reserve(static_cast<std::size_t>(command_rows.rows));
    captured.command_bindings.reserve(
        static_cast<std::size_t>(binding_rows.rows));
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (captured.commands.capacity() != command_rows.rows ||
      captured.command_bindings.capacity() != binding_rows.rows) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  captured.command_capacity = static_cast<std::size_t>(command_rows.rows);
  captured.binding_capacity = static_cast<std::size_t>(binding_rows.rows);
  captured.parameter_capacity = static_cast<std::size_t>(parameter_rows.rows);
  captured.producer_binding_slot_upper =
      static_cast<NSUInteger>(limit.backend_command_binding_slot_upper);
  encoder = [[RUNDMetalPipelineCapture alloc] initWithCapture:&captured];
  captured.guard_zero = pipeline->guard_zero;
  captured.guard_states = pipeline->states;
  captured.guard_state_count = pipeline->state_count;
  if (aggregate_selected) {
    captured.unguarded = true;
    reset_command_count = 0u;
    id<MTLBuffer> const step_control =
        profile_steps ? pipeline->step_control : pipeline->control;
    const rund::AccelCheck encoded = EncodeMetalNestedAggregate(
        native_aggregate, pipeline->control, step_control, encoder);
    if (!encoded.ok) {
      return encoded;
    }
    const rund::AccelCheck capture = CheckMetalPipelineCapture(captured);
    if (!capture.ok) {
      return capture;
    }
    if (captured.commands.size() == 2u) {
      captured.commands.back().control = true;
    }
    return rund::AccelCheck{true, "ok"};
  }
  MetalPipelineStatusParams open = status_params;
  open.phase = 0u;
  [encoder setComputePipelineState:complete];
  [encoder setBuffer:pipeline->control offset:0u atIndex:0u];
  [encoder setBytes:&open length:sizeof(open) atIndex:1u];
  if (profile_steps) {
    [encoder setBuffer:pipeline->step_control offset:0u atIndex:2u];
  }
  id<MTLBuffer> const states =
      pipeline->states == nil ? pipeline->control : pipeline->states;
  [encoder setBuffer:states offset:0u atIndex:3u];
  const NSUInteger count = std::max<NSUInteger>(pipeline->state_count, 1u);
  const NSUInteger width =
      std::min(count, [complete maxTotalThreadsPerThreadgroup]);
  [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
      threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
  const rund::AccelCheck opened = CheckMetalPipelineCapture(captured);
  if (!opened.ok) {
    return opened;
  }
  captured.commands.back().control = true;
  [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  if (private_raw_count != 0u) {
    [encoder setComputePipelineState:reset];
    [encoder setBuffer:pipeline->raw_status offset:0u atIndex:0u];
    [encoder setBytes:status_resets.data()
               length:status_resets.size() * sizeof(MetalPipelineResetMeta)
              atIndex:1u];
    [encoder setBytes:&status_params length:sizeof(status_params) atIndex:2u];
    const NSUInteger reset_threads = private_raw_count;
    const NSUInteger reset_width =
        std::min(reset_threads, [reset maxTotalThreadsPerThreadgroup]);
    [encoder dispatchThreads:MTLSizeMake(reset_threads, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(reset_width, 1u, 1u)];
    const rund::AccelCheck reset_capture = CheckMetalPipelineCapture(captured);
    if (!reset_capture.ok) {
      return reset_capture;
    }
    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  return EncodePrograms();
}

#endif

} // namespace rund::node::accel::detail
