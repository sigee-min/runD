#include "../build.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck MetalPipelineBuild::Allocate(std::shared_ptr<void> &prepared,
                                              PreparedPipelineMemory &memory) {
  device = (__bridge id<MTLDevice>)pipeline->adapter->device.get();
  if (pipeline->dispatch_count == 0u) {
    pipeline->retained_bytes = sizeof(MetalSequence);
    memory.host = PreparedMemory{.current = pipeline->retained_bytes,
                                 .peak = pipeline->retained_bytes,
                                 .cumulative = pipeline->retained_bytes,
                                 .budget = pipeline->retained_bytes};
    prepared = std::static_pointer_cast<void>(pipeline);
    finished = true;
    return rund::AccelCheck{true, "ok"};
  }
  pipeline->uses_status_arena = status.status_entry_count != 0u;
  if (pipeline->uses_status_arena) {
    const NSUInteger raw_status_bytes =
        static_cast<NSUInteger>(raw_status_count) * sizeof(std::uint32_t);
    pipeline->raw_status =
        [device newBufferWithLength:raw_status_bytes
                            options:MTLResourceStorageModePrivate];
  }
  pipeline->control = [device newBufferWithLength:PreparedPipelineControlBytes
                                          options:MTLResourceStorageModeShared];
  // Window selectors need a stable buffer identity while commands are
  // captured. The exact range count is known only after capture and is
  // allocated in Finalize; one placeholder is sufficient here.
  pipeline->range_capacity =
      static_cast<std::uint32_t>(!native_windows.empty());
  if (pipeline->range_capacity != 0u) {
    pipeline->range_buffer =
        [device newBufferWithLength:sizeof(MetalRange)
                            options:MTLResourceStorageModeShared];
    pipeline->range_owners =
        [device newBufferWithLength:sizeof(std::uint32_t)
                            options:MTLResourceStorageModeShared];
  }
  if (pipeline->state_count != 0u) {
    pipeline->states = [device
        newBufferWithLength:static_cast<NSUInteger>(pipeline->state_count) *
                            sizeof(ResidentState)
                    options:MTLResourceStorageModeShared];
  }
  if (profile_steps) {
    const NSUInteger step_control_bytes =
        static_cast<NSUInteger>(status.declared_step_count) *
        PreparedPipelineStepControlBytes;
    pipeline->step_control =
        [device newBufferWithLength:step_control_bytes
                            options:MTLResourceStorageModeShared];
  }
  if ((pipeline->uses_status_arena && pipeline->raw_status == nil) ||
      pipeline->control == nil || [pipeline->control contents] == nullptr ||
      (pipeline->range_capacity != 0u &&
       (pipeline->range_buffer == nil ||
        [pipeline->range_buffer contents] == nullptr ||
        pipeline->range_owners == nil ||
        [pipeline->range_owners contents] == nullptr)) ||
      (pipeline->state_count != 0u &&
       (pipeline->states == nil || [pipeline->states contents] == nullptr)) ||
      (profile_steps && (pipeline->step_control == nil ||
                         [pipeline->step_control contents] == nullptr))) {
    return rund::AccelCheck{false, "accel_metal_buffer_failed"};
  }
  const PreparedPipelineControl initial{};
  std::memcpy([pipeline->control contents], &initial, sizeof(initial));
  if (pipeline->range_capacity != 0u) {
    std::memset([pipeline->range_buffer contents], 0, sizeof(MetalRange));
    *static_cast<std::uint32_t *>([pipeline->range_owners contents]) =
        std::numeric_limits<std::uint32_t>::max();
  }
  if (pipeline->state_count != 0u) {
    std::memset([pipeline->states contents], 0,
                static_cast<std::size_t>(pipeline->state_count) *
                    sizeof(ResidentState));
  }
  if (profile_steps) {
    auto *const controls = static_cast<PreparedPipelineStepControl *>(
        [pipeline->step_control contents]);
    for (std::size_t index = 0u; index < status.declared_step_count; ++index) {
      controls[index] = PreparedPipelineStepControl{};
    }
  }
  needs_import =
      std::any_of(status_bindings.begin(), status_bindings.end(),
                  [](const MetalPipelineStatusBindingRecord &record) {
                    return !record.binding.replace;
                  });
  needs_reset = private_raw_count != 0u;
  if (!PrepareMetalStatus(
          *pipeline->adapter, pipeline->uses_status_arena, needs_reset,
          needs_import, !pipeline->telemetry.empty(), profile_steps,
          reset_owner, import_owner, reduce_owner, complete_owner,
          telemetry_owner, !native_publications.empty(), publish_owner,
          !native_windows.empty(), advance_owner, !native_windows.empty(),
          gate_owner)) {
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  reset = (__bridge id<MTLComputePipelineState>)reset_owner.get();
  import = (__bridge id<MTLComputePipelineState>)import_owner.get();
  reduce = (__bridge id<MTLComputePipelineState>)reduce_owner.get();
  complete = (__bridge id<MTLComputePipelineState>)complete_owner.get();
  telemetry = (__bridge id<MTLComputePipelineState>)telemetry_owner.get();
  publish = (__bridge id<MTLComputePipelineState>)publish_owner.get();
  advance = (__bridge id<MTLComputePipelineState>)advance_owner.get();
  gate = (__bridge id<MTLComputePipelineState>)gate_owner.get();
  pipeline->gate = gate;
  if ((pipeline->uses_status_arena &&
       (reduce == nil || [reduce maxTotalThreadsPerThreadgroup] <
                             kMetalPipelineReductionWidth)) ||
      (needs_reset && reset == nil) || (needs_import && import == nil) ||
      complete == nil || (!pipeline->telemetry.empty() && telemetry == nil) ||
      (!native_publications.empty() && publish == nil) ||
      (!native_windows.empty() && (advance == nil || gate == nil))) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }

  const std::uint32_t status_reset_count =
      static_cast<std::uint32_t>(status_resets.size());
  status_params = MetalPipelineStatusParams{
      .reset_range_count = status_reset_count,
      .status_count = status.status_entry_count,
      .reset_word_count = private_raw_count,
      .declared_step_count = status.declared_step_count,
      .invalid_reason =
          static_cast<std::uint32_t>(rund::compute::Reason::ReasonInvalid),
      .generation_stride = status.generation_stride,
      .source_count = static_cast<std::uint32_t>(status_sources.size()),
      .phase = 1u,
  };

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
