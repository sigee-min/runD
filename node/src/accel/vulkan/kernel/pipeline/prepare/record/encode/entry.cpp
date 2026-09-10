#include "local.hpp"

#include "../../../../../command/capture.hpp"
#include "../../../../../map/api.hpp"
#include "../../../../../map/local.hpp"
#include "../../../telemetry.hpp"

#include <limits>

namespace rund::node::accel::detail::vulkan_record_detail {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {
[[nodiscard]] BackendBatchEntry
Route(const VulkanPipelineRecordEntry &entry) noexcept {
  return BackendBatchEntry{
      .run = entry.run,
      .prepared = &entry.prepared,
      .recurrence = {.window = entry.window ? &*entry.window : nullptr},
      .transducer = entry.transducer,
      .template_index = entry.template_index,
      .occurrence_index = entry.occurrence_index,
  };
}

} // namespace

rund::AccelCheck EncodeEntry(const Encoding &context, std::size_t index,
                             bool first, bool &scratch_seen) noexcept {
  auto &pipeline = context.pipeline;
  auto &recipe = context.recipe;
  const auto recording = context.recording;
  auto &capture = context.capture;
  auto *const failure = context.failure;
  const auto *const slice = context.slice;
  const VulkanPipelineRecordEntry &entry = recipe.entries[index];
  const BackendBatchEntry route = Route(entry);
  if (failure != nullptr) {
    failure->occurrence_route(route);
  }
  const std::uint32_t template_index = entry.template_index;
  auto *const resources =
      static_cast<VulkanKernelResources *>(entry.prepared.get());
  const bool transduced = entry.transducer != NoTileTransducer;
  const std::shared_ptr<void> *const transducer_resource =
      transduced && entry.transducer < pipeline.transducers.size()
          ? &pipeline.transducers[entry.transducer]
          : nullptr;
  if (resources == nullptr || template_index >= recipe.template_count ||
      (transduced &&
       (entry.transducer >= recipe.transducer_ready.size() ||
        !recipe.transducer_ready[entry.transducer] ||
        transducer_resource == nullptr || *transducer_resource == nullptr))) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  if (!first && (recipe.barriers[index] != 0u ||
                 (scratch_seen && resources->shared_scratch))) {
    EncodeVulkanComputeToComputeBarrier(recording);
  }
  scratch_seen = scratch_seen || resources->shared_scratch;
  const BackendWindow *const resident_window =
      entry.window ? &*entry.window : nullptr;
  const std::uint32_t failed_outer_window =
      resident_window != nullptr && resident_window->nested()
          ? resident_window->outer_iteration
          : PreparedPipelineNoStep;
  const std::uint32_t failed_inner_iteration =
      resident_window != nullptr &&
              resident_window->phase == BackendWindowPhase::NestedAction
          ? resident_window->inner_iteration
          : PreparedPipelineNoStep;
  std::uint32_t failed_nested_phase = PipelineNestedPhaseNoneCode;
  if (resident_window != nullptr) {
    rund::compute::PipelineNestedPhase public_phase{};
    if (!resident_window->nested_phase(public_phase) ||
        !EncodePipelineNestedPhase(public_phase, failed_nested_phase)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }
  if (resident_window != nullptr &&
      resident_window->phase == BackendWindowPhase::NestedSeed &&
      !Trace(context.timestamps, [&] {
        return EncodeVulkanWindow(recording, pipeline.window,
                                  static_cast<std::uint32_t>(index), true);
      })) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  const PreparedProgramStatusSlice status_range =
      recipe.status_ranges[template_index];
  const PreparedProgramStatusSlice telemetry_range =
      recipe.telemetry_ranges[template_index];
  const std::size_t status_step_end =
      static_cast<std::size_t>(status_range.first) + status_range.count;
  const std::size_t telemetry_step_end =
      static_cast<std::size_t>(telemetry_range.first) + telemetry_range.count;
  if (status_range.count != resources->size() ||
      telemetry_range.count != resources->size() ||
      status_step_end > recipe.status_steps.size() ||
      telemetry_step_end > recipe.telemetry_steps.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  if (transduced) {
    const PreparedProgramStatusSlice program_status =
        recipe.status.slices[template_index];
    const bool status_empty =
        status_range.count == 1u &&
        recipe.status_steps[status_range.first].count == 0u;
    const bool telemetry_empty =
        telemetry_range.count == 1u &&
        recipe.telemetry_steps[telemetry_range.first].count == 0u;
    if (resident_window == nullptr ||
        resident_window->phase != BackendWindowPhase::NestedAction ||
        resident_window->inner_advance != 0u || resources->size() != 1u ||
        program_status.count != 0u || !status_empty || !telemetry_empty) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    rund::AccelCheck encoded{};
    {
      VulkanDispatchScope scope{capture, recording, resident_window->state};
      encoded = Trace(context.timestamps, [&] {
        return EncodeVulkanMap(*pipeline.adapter, *transducer_resource,
                               reinterpret_cast<void *>(recording));
      });
    }
    if (!encoded.ok) {
      return encoded;
    }
    if (slice != nullptr && slice->capture != nullptr &&
        slice->capture->failed) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (capture.failed) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  } else {
    for (std::size_t step_index = 0u; step_index < resources->size();
         ++step_index) {
      if (failure != nullptr) {
        failure->node_route(route, step_index);
      }
      VulkanKernelEntry *const step = resources->entry(step_index);
      if (step == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      const rund::AccelCheck encoded = Trace(context.timestamps, [&] {
        const auto encode_step = [&] {
          const rund::AccelCheck reset =
              EncodeVulkanResets(*resources, step_index, recording);
          return reset.ok
                     ? EncodeVulkanStep(*pipeline.adapter, *step, recording)
                     : reset;
        };
        if (resident_window == nullptr) {
          if (slice != nullptr && slice->capture != nullptr) {
            VulkanDispatchScope scope{*slice->capture, recording,
                                      slice->capture_owner};
            return encode_step();
          }
          return encode_step();
        }
        VulkanDispatchScope scope{capture, recording, resident_window->state};
        return encode_step();
      });
      if (!encoded.ok) {
        return encoded;
      }
      if (slice != nullptr && slice->capture != nullptr &&
          slice->capture->failed) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      if (resident_window != nullptr && capture.failed) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      const rund::AccelCheck observed = EncodeEvidence(
          context,
          StepEvidence{template_index, status_range, telemetry_range,
                       failed_outer_window, failed_inner_iteration,
                       failed_nested_phase,
                       resident_window == nullptr
                           ? std::numeric_limits<std::uint32_t>::max()
                           : resident_window->state},
          step_index);
      if (!observed.ok) {
        return observed;
      }
    }
  }
  if (failure != nullptr) {
    failure->occurrence_route(route);
  }
  const PreparedProgramStatusSlice program_status =
      recipe.status.slices[template_index];
  std::uint64_t folded = 0u;
  for (std::size_t step_index = 0u; step_index < status_range.count;
       ++step_index) {
    const PreparedProgramStatusSlice slice =
        recipe.status_steps[status_range.first + step_index];
    if (slice.count != 0u) {
      const VulkanPipelineCanonicalStatus &current =
          recipe.canonical[slice.first];
      folded += current.source.count;
    }
  }
  if (folded != program_status.count) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  if (!Trace(context.timestamps, [&] {
        return EncodeVulkanWindow(recording, pipeline.window,
                                  static_cast<std::uint32_t>(index));
      })) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (resident_window != nullptr &&
      resident_window->phase == BackendWindowPhase::NestedFold &&
      !Trace(context.timestamps, [&] {
        return EncodeVulkanPipelineWindowPublish(
            recording, pipeline.publish, resident_window->state,
            resident_window->outer_iteration);
      })) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (resident_window != nullptr && resident_window->advances_outer_state() &&
      resident_window->outer_iteration + 1u == resident_window->outer_bound &&
      !Trace(context.timestamps, [&] {
        return EncodeVulkanPipelineCanonicalize(recording, pipeline.publish,
                                                resident_window->state);
      })) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  return rund::AccelCheck{true, ""};
}

#endif
} // namespace rund::node::accel::detail::vulkan_record_detail
