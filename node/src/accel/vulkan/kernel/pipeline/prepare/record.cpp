#include "record.hpp"

#include "../../../../kernel/backend/exception.hpp"
#include "../../../../kernel/backend/pipeline_failure.hpp"
#include "../../../../kernel/prepared/template_registry.hpp"
#include "../../../../kernel/recurrence.hpp"
#include "../../../command.hpp"
#include "../../../command/resources.hpp"
#include "../../../map/api.hpp"
#include "../../../map/local.hpp"
#include "../../lease.hpp"
#include "../telemetry.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] BackendBatchEntry
Route(const VulkanPipelineRecordEntry &entry) noexcept {
  return BackendBatchEntry{
      .prepared = &entry.prepared,
      .recurrence = {.window = entry.window ? &*entry.window : nullptr},
      .transducer = entry.transducer,
      .template_index = entry.template_index,
      .occurrence_index = entry.occurrence_index,
  };
}

template <class T>
[[nodiscard]] std::uint64_t CapacityBytes(const std::vector<T> &values) {
  return ::rund::detail::counter::SaturatingMultiply(
      static_cast<std::uint64_t>(values.capacity()), sizeof(T));
}

} // namespace

rund::AccelCheck MakeVulkanPipelineRecordRecipe(
    const std::span<const BackendBatchEntry> entries,
    const std::span<const std::uint8_t> barriers,
    const std::span<const TileTransducer> transducers,
    const std::span<const VulkanPipelineCanonicalStatus> canonical,
    const std::span<const PreparedProgramStatusSlice> status_steps,
    const std::span<const PreparedProgramStatusSlice> telemetry_steps,
    const std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
        &status_ranges,
    const std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
        &telemetry_ranges,
    const PreparedPipelineStatusLayout &status,
    const std::size_t template_count, const std::uint64_t window_dispatches,
    const std::uint64_t window_gate_count, const bool recurrence,
    VulkanPipelineRecordRecipe &recipe) noexcept {
  try {
    recipe = {};
    recipe.entries.reserve(entries.size());
    for (const BackendBatchEntry &entry : entries) {
      if (entry.prepared == nullptr || *entry.prepared == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      recipe.entries.push_back(VulkanPipelineRecordEntry{
          .prepared = *entry.prepared,
          .window =
              entry.recurrence.window == nullptr
                  ? std::optional<BackendWindow>{}
                  : std::optional<BackendWindow>{*entry.recurrence.window},
          .transducer = entry.transducer,
          .template_index = entry.template_index,
          .occurrence_index = entry.occurrence_index,
      });
    }
    recipe.barriers.assign(barriers.begin(), barriers.end());
    recipe.transducer_ready.reserve(transducers.size());
    for (const TileTransducer &transducer : transducers) {
      recipe.transducer_ready.push_back(transducer.recurrence.ready());
    }
    recipe.canonical.assign(canonical.begin(), canonical.end());
    recipe.status_steps.assign(status_steps.begin(), status_steps.end());
    recipe.telemetry_steps.assign(telemetry_steps.begin(),
                                  telemetry_steps.end());
    recipe.status_ranges = status_ranges;
    recipe.telemetry_ranges = telemetry_ranges;
    recipe.status = status;
    recipe.template_count = template_count;
    recipe.window_dispatches = window_dispatches;
    recipe.window_gate_count = window_gate_count;
    recipe.recurrence = recurrence;
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    recipe = {};
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return recipe.entries.size() == entries.size() &&
                 recipe.barriers.size() == barriers.size()
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_pipeline_capacity"};
}

rund::AccelCheck
RecordVulkanPipeline(VulkanPipeline &pipeline, VulkanCommand &command,
                     const CommandKind kind, const bool replay,
                     PreparedPipelineFailureContext *const failure,
                     VulkanTimestampCapture *const timestamps) noexcept {
  if (pipeline.adapter == nullptr || pipeline.record == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
  }
  VulkanPipelineRecordRecipe &recipe = *pipeline.record;
  const rund::AccelCheck begun =
      BeginCommand(pipeline.adapter->device, command, kind);
  if (!begun.ok) {
    return begun;
  }
  VulkanLeaseScope recording_leases{*pipeline.adapter,
                                    pipeline.window.descriptor_leases};
  const VkCommandBuffer recording = command.buffer;
  if (timestamps != nullptr) {
    vkCmdResetQueryPool(recording, timestamps->queries, 0u,
                        timestamps->capacity);
  }
  const auto traced = [timestamps](auto &&encode) {
    if (timestamps == nullptr) {
      return encode();
    }
    VulkanTimestampScope scope{*timestamps};
    return encode();
  };
  if (!OpenVulkanPipelineControl(recording, pipeline.control)) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (pipeline.profile != nullptr) {
    if (!ResetVulkanPipelineProfile(recording, pipeline.control)) {
      return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
    }
  }
  if (!pipeline.window.routes.empty() &&
      !EncodeVulkanWindowStart(recording, pipeline.window)) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  if (recipe.recurrence) {
    const rund::AccelCheck encoded = traced([&] {
      return EncodeVulkanMap(*pipeline.adapter, pipeline.recurrence,
                             reinterpret_cast<void *>(recording));
    });
    if (!encoded.ok) {
      return encoded;
    }
  }
  VulkanDispatchCapture replay_capture = pipeline.window.capture;
  if (replay) {
    replay_capture.cursor = 0u;
    replay_capture.indirect_count = 0u;
    replay_capture.failed = false;
    replay_capture.replay = true;
  }
  VulkanDispatchCapture &capture =
      replay ? replay_capture : pipeline.window.capture;
  bool scratch_seen = false;
  for (std::size_t index = 0u;
       !recipe.recurrence && index < recipe.entries.size(); ++index) {
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
    if (index != 0u && (recipe.barriers[index] != 0u ||
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
        !traced([&] {
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
        encoded = traced([&] {
          return EncodeVulkanMap(*pipeline.adapter, *transducer_resource,
                                 reinterpret_cast<void *>(recording));
        });
      }
      if (!encoded.ok) {
        return encoded;
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
        const rund::AccelCheck encoded = traced([&] {
          rund::AccelCheck current{};
          if (resident_window == nullptr) {
            current = EncodeVulkanResets(*resources, step_index, recording);
            if (current.ok) {
              current = EncodeVulkanStep(*pipeline.adapter, *step, recording);
            }
          } else {
            VulkanDispatchScope scope{capture, recording,
                                      resident_window->state};
            current = EncodeVulkanResets(*resources, step_index, recording);
            if (current.ok) {
              current = EncodeVulkanStep(*pipeline.adapter, *step, recording);
            }
          }
          return current;
        });
        if (!encoded.ok) {
          return encoded;
        }
        if (resident_window != nullptr && capture.failed) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        const PreparedProgramStatusSlice status_slice =
            recipe.status_steps[status_range.first + step_index];
        const std::size_t status_end =
            static_cast<std::size_t>(status_slice.first) + status_slice.count;
        if (status_end > recipe.canonical.size()) {
          return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
        }
        if (status_slice.count != 0u) {
          EncodeVulkanComputeToComputeBarrier(recording);
        }
        for (std::size_t status_index = status_slice.first;
             status_index < status_end; ++status_index) {
          const VulkanPipelineCanonicalStatus &current =
              recipe.canonical[status_index];
          if (current.active_program != template_index) {
            return rund::AccelCheck{false,
                                    "accel_kernel_primitive_unsupported"};
          }
          if (!EncodeVulkanPipelineCanonicalStatus(recording, pipeline.control,
                                                   current) ||
              !FoldVulkanPipelineControl(
                  recording, pipeline.control,
                  PreparedProgramStatusSlice{.first = current.first,
                                             .count = current.source.count},
                  recipe.status.declared_steps[template_index],
                  failed_outer_window, failed_inner_iteration,
                  failed_nested_phase)) {
            return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
          }
        }
        const PreparedProgramStatusSlice telemetry_slice =
            recipe.telemetry_steps[telemetry_range.first + step_index];
        const std::size_t telemetry_end =
            static_cast<std::size_t>(telemetry_slice.first) +
            telemetry_slice.count;
        if (telemetry_end > pipeline.telemetry.size() ||
            !EncodeVulkanTelemetry(
                pipeline, recording,
                std::span<const VulkanPipelineTelemetryRecord>{
                    pipeline.telemetry}
                    .subspan(telemetry_slice.first, telemetry_slice.count),
                resident_window == nullptr
                    ? std::numeric_limits<std::uint32_t>::max()
                    : resident_window->state,
                status_slice.count == 0u)) {
          return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
        }
        if (status_slice.count != 0u && telemetry_slice.count == 0u) {
          EncodeVulkanComputeToComputeBarrier(recording);
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
    if (!traced([&] {
          return EncodeVulkanWindow(recording, pipeline.window,
                                    static_cast<std::uint32_t>(index));
        })) {
      return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
    }
    if (resident_window != nullptr &&
        resident_window->phase == BackendWindowPhase::NestedFold &&
        !traced([&] {
          return EncodeVulkanPipelineWindowPublish(
              recording, pipeline.publish, resident_window->state,
              resident_window->outer_iteration);
        })) {
      return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
    }
    if (resident_window != nullptr && resident_window->advances_outer_state() &&
        resident_window->outer_iteration + 1u == resident_window->outer_bound &&
        !traced([&] {
          return EncodeVulkanPipelineCanonicalize(recording, pipeline.publish,
                                                  resident_window->state);
        })) {
      return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
    }
  }
  if (failure != nullptr) {
    failure->stage(PreparedPipelineFailureStage::BackendFinalization);
  }
  if (capture.failed) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (capture.cursor != recipe.window_dispatches ||
      capture.indirect_count != recipe.window_gate_count ||
      pipeline.window.gates.size() != recipe.window_gate_count) {
    return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  if (!replay && !FreezeVulkanWindow(pipeline.window)) {
    return rund::AccelCheck{false, "accel_vulkan_memory_unavailable"};
  }
  if (!FinishVulkanPipelineControl(recording, pipeline.control,
                                   recipe.status) ||
      !traced([&] {
        return EncodeVulkanPipelinePublish(recording, pipeline.publish);
      }) ||
      !PublishVulkanPipelineControl(recording, pipeline.control)) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  return EndCommand(command);
}

std::uint64_t VulkanPipelineRecordHostBytes(
    const VulkanPipelineRecordRecipe &recipe) noexcept {
  std::uint64_t bytes = sizeof(VulkanPipelineRecordRecipe);
  for (const std::uint64_t current :
       {CapacityBytes(recipe.entries), CapacityBytes(recipe.barriers),
        CapacityBytes(recipe.transducer_ready), CapacityBytes(recipe.canonical),
        CapacityBytes(recipe.status_steps),
        CapacityBytes(recipe.telemetry_steps)}) {
    bytes = ::rund::detail::counter::SaturatingAdd(bytes, current);
  }
  return bytes;
}

bool VulkanPipelineRecordHostBytes(
    const PreparedKernelPipelineReservation &reservation,
    std::uint64_t &bytes) noexcept {
  bytes = sizeof(VulkanPipelineRecordRecipe);
  const auto add_extent = [&bytes](const std::uint64_t count,
                                   const std::uint64_t element_bytes) noexcept {
    std::uint64_t extent = 0u;
    return rund::kernel::checked::mul(count, element_bytes, extent) &&
           rund::kernel::checked::add(bytes, extent, bytes);
  };
  return add_extent(reservation.occurrence_count,
                    sizeof(VulkanPipelineRecordEntry)) &&
         add_extent(reservation.occurrence_count, sizeof(std::uint8_t)) &&
         add_extent(reservation.nested_group_count, sizeof(std::uint8_t)) &&
         add_extent(reservation.backend_status_source_count,
                    sizeof(VulkanPipelineCanonicalStatus)) &&
         add_extent(reservation.backend_step_description_count,
                    sizeof(PreparedProgramStatusSlice)) &&
         add_extent(reservation.backend_step_description_count,
                    sizeof(PreparedProgramStatusSlice));
}

#endif

} // namespace rund::node::accel::detail
