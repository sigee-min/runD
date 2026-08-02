#include "../build.hpp"

#include "../aggregate/prepare.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck PopulateAggregateProfile(
    const NestedAggregate &aggregate,
    const std::span<const BackendBatchEntry> templates,
    const PreparedPipelineStatusLayout &status, const std::uint32_t owner,
    const std::uint64_t work_items, const std::uint64_t workgroups,
    const std::uint64_t physical_dispatches,
    std::vector<PreparedPipelineStepEvidence> &rows,
    PreparedPipelineFailureContext &failure_context) noexcept {
  failure_context.clear_route();
  if (templates.size() != status.active_step_count ||
      rows.size() != status.declared_step_count || owner >= rows.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::fill(rows.begin(), rows.end(), PreparedPipelineStepEvidence{});
  for (std::size_t index = 0u; index < templates.size(); ++index) {
    failure_context.template_route(static_cast<std::uint32_t>(index));
    const BackendRun *const run = templates[index].run;
    const std::uint32_t declared = status.declared_steps[index];
    const std::uint64_t occurrences = aggregate.authored_occurrences(index);
    if (run == nullptr || declared >= rows.size() || occurrences == 0u ||
        run->original_dispatch_count == 0u ||
        occurrences > std::numeric_limits<std::uint64_t>::max() /
                          run->original_dispatch_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const std::uint64_t original = occurrences * run->original_dispatch_count;
    PreparedPipelineStepEvidence &row = rows[declared];
    if (original > std::numeric_limits<std::uint64_t>::max() -
                       row.original_dispatch_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    row.original_dispatch_count += original;
  }
  PreparedPipelineStepEvidence &physical = rows[owner];
  physical.final_dispatch_count = physical_dispatches;
  physical.physical_dispatch_count = physical_dispatches;
  physical.workgroup_count = workgroups;
  physical.work_item_count = work_items;
  return rund::AccelCheck{true, "ok"};
}

} // namespace

rund::AccelCheck MetalPipelineBuild::Allocate(std::shared_ptr<void> &prepared,
                                              PreparedPipelineMemory &memory) {
  device = (__bridge id<MTLDevice>)pipeline->adapter->device.get();
  if (aggregate_selected) {
    pipeline->dispatch_count = 2u;
    pipeline->reset_count = 0u;
    pipeline->reset_bytes = 0u;
    pipeline->state_count = 0u;
    pipeline->uses_status_arena = false;
    pipeline->direct_aggregate = true;
    pipeline->recurrence.reset();
    pipeline->transducers = {};
    pipeline->telemetry = {};
    pipeline->control =
        [device newBufferWithLength:PreparedPipelineControlBytes
                            options:MTLResourceStorageModeShared];
    if (profile_steps) {
      pipeline->step_control =
          [device newBufferWithLength:static_cast<NSUInteger>(
                                          status.declared_step_count) *
                                      PreparedPipelineStepControlBytes
                              options:MTLResourceStorageModeShared];
    }
    if (pipeline->control == nil || [pipeline->control contents] == nullptr ||
        (profile_steps &&
         (pipeline->step_control == nil ||
          [pipeline->step_control contents] == nullptr ||
          aggregate_profile_owner >= pipeline->step_evidence.size()))) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
    const PreparedPipelineControl initial{};
    std::memcpy([pipeline->control contents], &initial, sizeof(initial));
    const rund::AccelCheck aggregate_ready =
        PrepareMetalNestedAggregate(*pipeline->adapter, native_aggregate);
    if (!aggregate_ready.ok) {
      return aggregate_ready;
    }
    if (profile_steps) {
      auto *const controls = static_cast<PreparedPipelineStepControl *>(
          [pipeline->step_control contents]);
      std::fill_n(controls, pipeline->step_evidence.size(),
                  PreparedPipelineStepControl{});
      const rund::AccelCheck projected = PopulateAggregateProfile(
          aggregates.front(), templates, status, aggregate_profile_owner,
          native_aggregate.work_item_count, native_aggregate.workgroup_count,
          pipeline->dispatch_count, pipeline->step_evidence, failure_context);
      if (!projected.ok) {
        return projected;
      }
    }
    return aggregate_ready;
  }
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
  pipeline->guard_zero =
      [device newBufferWithLength:sizeof(std::uint32_t)
                          options:MTLResourceStorageModeShared];
  if (pipeline->state_count != 0u) {
    pipeline->states = [device
        newBufferWithLength:static_cast<NSUInteger>(pipeline->state_count) *
                            sizeof(ResidentState)
                    options:MTLResourceStorageModePrivate];
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
      pipeline->guard_zero == nil ||
      [pipeline->guard_zero contents] == nullptr ||
      (pipeline->state_count != 0u && pipeline->states == nil) ||
      (profile_steps && (pipeline->step_control == nil ||
                         [pipeline->step_control contents] == nullptr))) {
    return rund::AccelCheck{false, "accel_metal_buffer_failed"};
  }
  const PreparedPipelineControl initial{};
  std::memcpy([pipeline->control contents], &initial, sizeof(initial));
  *static_cast<std::uint32_t *>([pipeline->guard_zero contents]) = 0u;
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
          telemetry_owner, native_publication_count != 0u, publish_owner,
          !native_windows.empty(), advance_owner)) {
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  reset = (__bridge id<MTLComputePipelineState>)reset_owner.get();
  import = (__bridge id<MTLComputePipelineState>)import_owner.get();
  reduce = (__bridge id<MTLComputePipelineState>)reduce_owner.get();
  complete = (__bridge id<MTLComputePipelineState>)complete_owner.get();
  telemetry = (__bridge id<MTLComputePipelineState>)telemetry_owner.get();
  publish = (__bridge id<MTLComputePipelineState>)publish_owner.get();
  advance = (__bridge id<MTLComputePipelineState>)advance_owner.get();
  if ((pipeline->uses_status_arena &&
       (reduce == nil || [reduce maxTotalThreadsPerThreadgroup] <
                             kMetalPipelineReductionWidth)) ||
      (needs_reset && reset == nil) || (needs_import && import == nil) ||
      complete == nil || (!pipeline->telemetry.empty() && telemetry == nil) ||
      (native_publication_count != 0u && publish == nil) ||
      (!native_windows.empty() && advance == nil)) {
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
      .state_count = pipeline->state_count,
  };

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
