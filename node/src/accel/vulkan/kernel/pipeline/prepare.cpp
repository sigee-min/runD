#include "state.hpp"

#include "../../command.hpp"
#include "../../command/resources.hpp"
#include "../../map/api.hpp"
#include "../../map/local.hpp"
#include "../../runtime/timestamp.hpp"
#include "../lease.hpp"
#include "evidence.hpp"
#include "recurrence.hpp"
#include "telemetry.hpp"

#include <rund/counter.hpp>

#include <array>
#include <limits>
#include <new>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck
PrepareVulkanPipeline(const std::span<const BackendBatchEntry> templates,
                      const std::span<const BackendBatchEntry> entries,
                      const std::span<const std::uint8_t> barriers,
                      const std::span<const BackendPublish> publications,
                      PreparedPipelineStatusLayout &status,
                      const bool profile_steps, std::shared_ptr<void> &prepared,
                      PreparedPipelineMemory &memory) {
  prepared.reset();
  memory = {};
  if (templates.empty() || entries.empty() ||
      entries.size() != barriers.size() ||
      templates.size() != status.active_step_count ||
      entries.size() != status.command_count) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  for (std::size_t index = 0u; index < entries.size(); ++index) {
    if (entries[index].occurrence_index != index ||
        entries[index].template_index >= templates.size() ||
        entries[index].run != templates[entries[index].template_index].run ||
        entries[index].prepared !=
            templates[entries[index].template_index].prepared) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
  }
  const MapRecurrence recurrence = BuildMapRecurrence(entries, barriers);
  if (recurrence.invalid()) {
    return rund::AccelCheck{false, recurrence.reason};
  }
  std::shared_ptr<VulkanPipeline> pipeline{};
  std::vector<VulkanPipelineCanonicalStatus> canonical{};
  std::vector<PreparedProgramStatusSlice> status_steps{};
  std::vector<PreparedProgramStatusSlice> telemetry_steps{};
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
      status_ranges{};
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
      telemetry_ranges{};
  std::array<VulkanPipelineWork, PreparedPipelineStepCapacity> template_work{};
  PreparedMemory recurrence_staging{};
  std::uint64_t window_dispatches = 0u;
  std::uint64_t status_command_sources = 0u;
  std::uint64_t telemetry_command_count = 0u;
  try {
    pipeline = std::make_shared<VulkanPipeline>();
    canonical.reserve(templates.size() * kInlineBoundStepCapacity);
    if (profile_steps) {
      pipeline->profile = std::make_unique<VulkanPipelineProfile>();
    }
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (pipeline->profile != nullptr) {
    if (status.active_step_count > PreparedPipelineStepCapacity ||
        status.declared_step_count > PreparedPipelineStepCapacity ||
        status.declared_step_count == 0u) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    pipeline->profile->active_step_count = status.active_step_count;
    pipeline->profile->command_count = status.command_count;
    pipeline->profile->declared_step_count = status.declared_step_count;
    pipeline->profile->declared_steps = status.declared_steps;
    std::array<bool, PreparedPipelineStepCapacity> declared{};
    for (std::size_t active = 0u; active < status.active_step_count; ++active) {
      const std::uint32_t ordinal = status.declared_steps[active];
      if (ordinal >= status.declared_step_count || declared[ordinal]) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      declared[ordinal] = true;
    }
  }
  if (recurrence.ready()) {
    const rund::AccelCheck ready = PrepareVulkanRecurrence(
        entries, recurrence, status, *pipeline, recurrence_staging);
    if (!ready.ok) {
      return ready;
    }
    if (pipeline->profile != nullptr) {
      const auto *const map = static_cast<const VulkanMapEncodeResources *>(
          pipeline->recurrence.get());
      VulkanPipelineWork work{};
      const bool exact = map != nullptr && AccumulateVulkanWork(*map, work) &&
                         work.dispatch_count == pipeline->dispatch_count;
      for (std::size_t index = 0u; index < entries.size(); ++index) {
        const BackendBatchEntry &entry = entries[index];
        const std::uint32_t template_index = entry.template_index;
        const std::uint32_t declared =
            template_index < status.active_step_count
                ? status.declared_steps[template_index]
                : PreparedPipelineNoStep;
        if (entry.run == nullptr ||
            template_index >= status.active_step_count ||
            declared >= status.declared_step_count) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
        const bool owner = index == 0u;
        pipeline->profile->rows[declared] = PreparedPipelineStepEvidence{
            .original_dispatch_count = entry.run->original_dispatch_count,
            .final_dispatch_count = owner ? pipeline->dispatch_count : 0u,
            .physical_dispatch_count = owner ? pipeline->dispatch_count : 0u,
            .workgroup_count = owner && exact ? work.workgroup_count : 0u,
            .work_item_count = owner && exact ? work.work_item_count : 0u,
        };
      }
    }
  }
  // Describe immutable status and telemetry once per compact route template.
  // Expanded physical occurrences reuse these slices during native capture.
  for (std::size_t template_index = 0u;
       !recurrence.ready() && template_index < templates.size();
       ++template_index) {
    const BackendBatchEntry &entry = templates[template_index];
    auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<VulkanKernelResources *>(entry.prepared->get());
    VulkanKernelContext context{};
    const rund::AccelCheck valid =
        entry.run == nullptr || entry.run->pick == nullptr
            ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
            : ValidateVulkanKernelContext(*entry.run->pick, context);
    if (!valid.ok || resources == nullptr || resources->size() == 0u ||
        !IsPipelinePrivatePreparation(resources->mode) ||
        resources->command.buffer != VK_NULL_HANDLE ||
        context.adapter == nullptr ||
        (pipeline->adapter != nullptr &&
         pipeline->adapter != context.adapter)) {
      return valid.ok ? rund::AccelCheck{false, "accel_kernel_run_invalid"}
                      : valid;
    }
    pipeline->adapter = context.adapter;
    if (entry.template_index != template_index ||
        status.declared_steps[template_index] >= status.declared_step_count) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }

    const std::size_t status_begin = status_steps.size();
    const std::size_t telemetry_begin = telemetry_steps.size();
    std::uint32_t entry_count = 0u;
    VulkanPipelineWork direct_work{};
    try {
      for (std::size_t step_index = 0u; step_index < resources->size();
           ++step_index) {
        VulkanKernelEntry *const step = resources->entry(step_index);
        if (step == nullptr || step->ops.pipeline_status == nullptr) {
          return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
        }
        if (pipeline->profile != nullptr && direct_work.exact &&
            !AccumulateVulkanWork(*step, direct_work)) {
          direct_work.exact = false;
          direct_work.dispatch_count = 0u;
          direct_work.workgroup_count = 0u;
          direct_work.work_item_count = 0u;
        }
        if (pipeline->telemetry.size() >
            std::numeric_limits<std::uint32_t>::max()) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        PreparedProgramStatusSlice telemetry_slice{
            .first = static_cast<std::uint32_t>(pipeline->telemetry.size())};
        if (step->ops.pipeline_telemetry != nullptr) {
          VulkanPipelineTelemetrySource telemetry{};
          const rund::AccelCheck telemetry_ready =
              step->ops.pipeline_telemetry(step->resource, telemetry);
          if (!telemetry_ready.ok) {
            return telemetry_ready;
          }
          if (telemetry.kind != VulkanPipelineTelemetryKind::None) {
            pipeline->telemetry.push_back(VulkanPipelineTelemetryRecord{
                .source = telemetry,
                .owner = step->resource,
                .declared_step = status.declared_steps[template_index],
            });
            telemetry_slice.count = 1u;
          }
        }
        telemetry_steps.push_back(telemetry_slice);
        VulkanPipelineStatusSource source{};
        const rund::AccelCheck described =
            step->ops.pipeline_status(step->resource, source);
        if (!described.ok) {
          return described;
        }
        if (canonical.size() > std::numeric_limits<std::uint32_t>::max()) {
          return rund::AccelCheck{false, "compute_pipeline_capacity"};
        }
        PreparedProgramStatusSlice status_slice{
            .first = static_cast<std::uint32_t>(canonical.size())};
        if (source.rule != VulkanPipelineStatusRule::None) {
          if (canonical.size() >= std::numeric_limits<std::uint32_t>::max() ||
              entry_count > std::numeric_limits<std::uint32_t>::max() -
                                status.status_entry_count ||
              source.count > std::numeric_limits<std::uint32_t>::max() -
                                 entry_count - status.status_entry_count) {
            return rund::AccelCheck{false, "compute_pipeline_capacity"};
          }
          canonical.push_back(VulkanPipelineCanonicalStatus{
              .source = source,
              .first = status.status_entry_count + entry_count,
              .active_program = static_cast<std::uint32_t>(template_index),
          });
          status_slice.count = 1u;
          entry_count += source.count;
        }
        status_steps.push_back(status_slice);
      }
      const std::size_t status_count = status_steps.size() - status_begin;
      const std::size_t telemetry_count =
          telemetry_steps.size() - telemetry_begin;
      if (status_begin > std::numeric_limits<std::uint32_t>::max() ||
          status_count > std::numeric_limits<std::uint32_t>::max() ||
          telemetry_begin > std::numeric_limits<std::uint32_t>::max() ||
          telemetry_count > std::numeric_limits<std::uint32_t>::max()) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      if (!SetPreparedProgramStatusSlice(
              status, static_cast<std::uint32_t>(template_index),
              entry_count)) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      status_ranges[template_index] = PreparedProgramStatusSlice{
          .first = static_cast<std::uint32_t>(status_begin),
          .count = static_cast<std::uint32_t>(status_count),
      };
      telemetry_ranges[template_index] = PreparedProgramStatusSlice{
          .first = static_cast<std::uint32_t>(telemetry_begin),
          .count = static_cast<std::uint32_t>(telemetry_count),
      };
      template_work[template_index] = direct_work;
    } catch (const std::bad_alloc &) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  for (std::size_t entry_index = 0u;
       !recurrence.ready() && entry_index < entries.size(); ++entry_index) {
    const BackendBatchEntry &entry = entries[entry_index];
    if (entry.template_index >= templates.size() ||
        entry.occurrence_index != entry_index) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<VulkanKernelResources *>(entry.prepared->get());
    if (resources == nullptr ||
        resources->dispatch_count > std::numeric_limits<std::uint64_t>::max() -
                                        pipeline->dispatch_count) {
      return rund::AccelCheck{false, resources == nullptr
                                         ? "accel_kernel_run_invalid"
                                         : "compute_pipeline_capacity"};
    }
    pipeline->dispatch_count += resources->dispatch_count;
    if (entry.recurrence.window != nullptr) {
      if (resources->dispatch_count >
          std::numeric_limits<std::uint64_t>::max() - window_dispatches) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      window_dispatches += resources->dispatch_count;
    }
    pipeline->reset_count = ::rund::detail::counter::SaturatingAdd(
        pipeline->reset_count, resources->reset_count);
    pipeline->reset_bytes = ::rund::detail::counter::SaturatingAdd(
        pipeline->reset_bytes, resources->reset_bytes);
    if (pipeline->profile != nullptr) {
      const std::uint32_t declared =
          status.declared_steps[entry.template_index];
      if (declared >= status.declared_step_count) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      PreparedPipelineStepEvidence &row = pipeline->profile->rows[declared];
      row.original_dispatch_count = ::rund::detail::counter::SaturatingAdd(
          row.original_dispatch_count, entry.run->original_dispatch_count);
      row.final_dispatch_count = ::rund::detail::counter::SaturatingAdd(
          row.final_dispatch_count, entry.run->final_dispatch_count);
      row.physical_dispatch_count = ::rund::detail::counter::SaturatingAdd(
          row.physical_dispatch_count, resources->dispatch_count);
      const VulkanPipelineWork &work = template_work[entry.template_index];
      if (work.exact && work.dispatch_count == resources->dispatch_count) {
        row.workgroup_count = ::rund::detail::counter::SaturatingAdd(
            row.workgroup_count, work.workgroup_count);
        row.work_item_count = ::rund::detail::counter::SaturatingAdd(
            row.work_item_count, work.work_item_count);
      }
    }
    const PreparedProgramStatusSlice status_range =
        status_ranges[entry.template_index];
    const PreparedProgramStatusSlice telemetry_range =
        telemetry_ranges[entry.template_index];
    for (std::size_t step = 0u; step < status_range.count; ++step) {
      status_command_sources = ::rund::detail::counter::SaturatingAdd(
          status_command_sources,
          status_steps[status_range.first + step].count);
    }
    for (std::size_t step = 0u; step < telemetry_range.count; ++step) {
      telemetry_command_count = ::rund::detail::counter::SaturatingAdd(
          telemetry_command_count,
          telemetry_steps[telemetry_range.first + step].count);
    }
    if (status_command_sources == std::numeric_limits<std::uint64_t>::max() ||
        telemetry_command_count == std::numeric_limits<std::uint64_t>::max()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }

  std::scoped_lock lock{pipeline->submission.mutex, pipeline->adapter->mutex};
  if (pipeline->dispatch_count == 0u) {
    const std::uint64_t bytes =
        sizeof(VulkanPipeline) +
        (pipeline->profile == nullptr ? 0u : sizeof(VulkanPipelineProfile));
    memory.host = PreparedMemory{
        .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
    prepared = std::static_pointer_cast<void>(pipeline);
    return rund::AccelCheck{true, "ok"};
  }
  for (std::size_t index = 0u; index < canonical.size(); ++index) {
    if (canonical[index].source.raw == nullptr ||
        canonical[index].source.raw->buffer == VK_NULL_HANDLE ||
        canonical[index].active_program >= status.active_step_count) {
      return FailVulkanPipeline(pipeline, "accel_kernel_primitive_unsupported");
    }
  }
  const rund::AccelCheck control_ready =
      PrepareVulkanPipelineControl(*pipeline->adapter, canonical, status,
                                   profile_steps, pipeline->control, memory);
  if (!control_ready.ok) {
    return FailVulkanPipeline(pipeline, control_ready.reason);
  }
  if (status_command_sources >
      (std::numeric_limits<std::uint64_t>::max() - 2u) / 2u) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  pipeline->control.command_count = 2u + 2u * status_command_sources;
  if (window_dispatches >
      std::numeric_limits<std::uint64_t>::max() - pipeline->dispatch_count) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  const std::uint64_t capture_capacity =
      pipeline->dispatch_count + window_dispatches;
  const rund::AccelCheck window_ready =
      PrepareVulkanWindow(*pipeline->adapter, entries, capture_capacity, status,
                          pipeline->control.summary, pipeline->window);
  if (!window_ready.ok) {
    return FailVulkanPipeline(pipeline, window_ready.reason);
  }
  const std::uint64_t window_bytes = ::rund::detail::counter::SaturatingAdd(
      pipeline->window.states.bytes,
      ::rund::detail::counter::SaturatingAdd(
          pipeline->window.arguments.bytes,
          ::rund::detail::counter::SaturatingAdd(
              pipeline->window.original_arguments.bytes,
              pipeline->window.owners.bytes)));
  accumulate_memory(memory.staging, PreparedMemory{.current = window_bytes,
                                                   .peak = window_bytes,
                                                   .cumulative = window_bytes,
                                                   .budget = window_bytes});
  const rund::AccelCheck publish_ready = PrepareVulkanPipelinePublish(
      *pipeline->adapter, publications, status, pipeline->control,
      pipeline->window, pipeline->publish);
  if (!publish_ready.ok) {
    return FailVulkanPipeline(pipeline, publish_ready.reason);
  }
  std::uint64_t seed_preflight_count = 0u;
  std::uint64_t seal_count = 0u;
  for (const VulkanWindowRoute &window : pipeline->window.routes) {
    const auto phase = static_cast<BackendWindowPhase>(window.params.phase);
    seed_preflight_count = ::rund::detail::counter::SaturatingAdd(
        seed_preflight_count,
        static_cast<std::uint64_t>(phase == BackendWindowPhase::NestedSeed));
    if (phase != BackendWindowPhase::Ordinary &&
        phase != BackendWindowPhase::NestedSeed) {
      continue;
    }
    for (const VulkanPipelinePublishRoute &publication :
         pipeline->publish.routes) {
      seal_count = ::rund::detail::counter::SaturatingAdd(
          seal_count, static_cast<std::uint64_t>(publication.params.state ==
                                                 window.params.state));
    }
  }
  const std::uint64_t control_dispatches =
      ::rund::detail::counter::SaturatingAdd(
          ::rund::detail::counter::SaturatingAdd(pipeline->window.routes.size(),
                                                 seed_preflight_count),
          ::rund::detail::counter::SaturatingAdd(
              pipeline->publish.routes.size(), seal_count));
  if (control_dispatches == std::numeric_limits<std::uint64_t>::max() ||
      control_dispatches > std::numeric_limits<std::uint64_t>::max() -
                               pipeline->dispatch_count) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  pipeline->dispatch_count += control_dispatches;
  accumulate_memory(memory.staging, recurrence_staging);
  if (!PrepareVulkanTelemetry(*pipeline)) {
    const char *const reason = VulkanLastError(pipeline->adapter);
    return FailVulkanPipeline(pipeline,
                              reason == nullptr || reason[0] == '\0'
                                  ? "accel_vulkan_descriptor_unavailable"
                                  : reason);
  }
  pipeline->control.command_count = ::rund::detail::counter::SaturatingAdd(
      pipeline->control.command_count, telemetry_command_count);
  if (pipeline->control.command_count ==
      std::numeric_limits<std::uint64_t>::max()) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  const rund::AccelCheck command_ready = CreateCommand(
      pipeline->adapter->device, pipeline->adapter->compute_queue_family,
      pipeline->command, CommandKind::ReusablePrimary);
  if (!command_ready.ok) {
    return FailVulkanPipeline(pipeline, command_ready.reason);
  }
  if (pipeline->profile != nullptr) {
    VulkanPipelineProfile &profile = *pipeline->profile;
    // Two field-major fills plus their transfer-to-compute dependency.
    profile.instrumentation_command_count = 3u;
    profile.instrumentation_byte_count =
        static_cast<std::uint64_t>(profile.declared_step_count) *
        PreparedPipelineStepControlBytes;
    if (VulkanTimestampAvailable(*pipeline->adapter)) {
      const std::uint64_t timestamp_commands =
          recurrence.ready() ? 1u : entries.size();
      if (timestamp_commands > std::numeric_limits<std::uint32_t>::max() / 2u) {
        return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
      }
      VkQueryPoolCreateInfo query{};
      query.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
      query.queryType = VK_QUERY_TYPE_TIMESTAMP;
      query.queryCount = static_cast<std::uint32_t>(2u * timestamp_commands);
      VkQueryPool timestamps = VK_NULL_HANDLE;
      if (query.queryCount != 0u &&
          vkCreateQueryPool(pipeline->adapter->device, &query, nullptr,
                            &timestamps) == VK_SUCCESS &&
          timestamps != VK_NULL_HANDLE) {
        try {
          profile.timestamped.resize(timestamp_commands);
          profile.command_templates.resize(timestamp_commands);
          profile.timestamp_values.resize(query.queryCount);
          if (recurrence.ready()) {
            profile.command_templates[0u] = 0u;
          } else {
            for (std::size_t index = 0u; index < entries.size(); ++index) {
              profile.command_templates[index] = entries[index].template_index;
            }
          }
        } catch (const std::bad_alloc &) {
          vkDestroyQueryPool(pipeline->adapter->device, timestamps, nullptr);
          return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
        }
        profile.timestamps = timestamps;
        profile.command_count = static_cast<std::uint32_t>(timestamp_commands);
        profile.query_count = query.queryCount;
        // One bulk reset and one timestamp at each side of every active
        // Program. Query storage is reported at its observable U64 width.
        profile.instrumentation_command_count += 1u + profile.query_count;
        profile.instrumentation_byte_count +=
            static_cast<std::uint64_t>(profile.query_count) *
            sizeof(std::uint64_t);
      } else if (timestamps != VK_NULL_HANDLE) {
        vkDestroyQueryPool(pipeline->adapter->device, timestamps, nullptr);
      }
    }
  }
  const rund::AccelCheck begun =
      BeginCommand(pipeline->adapter->device, pipeline->command,
                   CommandKind::ReusablePrimary);
  if (!begun.ok) {
    return FailVulkanPipeline(pipeline, begun.reason);
  }
  VulkanLeaseScope recording_leases{*pipeline->adapter,
                                    pipeline->window.descriptor_leases};
  const VkCommandBuffer recording = pipeline->command.buffer;
  if (!OpenVulkanPipelineControl(recording, pipeline->control)) {
    return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
  }
  if (pipeline->profile != nullptr) {
    if (!ResetVulkanPipelineProfile(recording, pipeline->control)) {
      return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
    }
    if (pipeline->profile->timestamps != VK_NULL_HANDLE) {
      vkCmdResetQueryPool(recording, pipeline->profile->timestamps, 0u,
                          pipeline->profile->query_count);
    }
  }
  if (!pipeline->window.routes.empty()) {
    if (!EncodeVulkanWindowStart(recording, pipeline->window)) {
      return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
    }
  }
  if (recurrence.ready()) {
    if (pipeline->profile != nullptr &&
        pipeline->profile->timestamps != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(recording, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          pipeline->profile->timestamps, 0u);
    }
    const rund::AccelCheck encoded =
        EncodeVulkanMap(*pipeline->adapter, pipeline->recurrence,
                        reinterpret_cast<void *>(recording));
    if (!encoded.ok) {
      return FailVulkanPipeline(pipeline, encoded.reason);
    }
    if (pipeline->profile != nullptr &&
        pipeline->profile->timestamps != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(recording, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          pipeline->profile->timestamps, 1u);
      pipeline->profile->timestamped[0u] = true;
    }
  }
  bool scratch_seen = false;
  for (std::size_t index = 0u; !recurrence.ready() && index < entries.size();
       ++index) {
    const std::uint32_t template_index = entries[index].template_index;
    auto *const resources =
        static_cast<VulkanKernelResources *>(entries[index].prepared->get());
    if (resources == nullptr || template_index >= templates.size()) {
      return FailVulkanPipeline(pipeline, "accel_kernel_run_invalid");
    }
    if (index != 0u && (barriers[index] != 0u ||
                        (scratch_seen && resources->shared_scratch))) {
      EncodeVulkanComputeToComputeBarrier(recording);
    }
    scratch_seen = scratch_seen || resources->shared_scratch;
    const BackendWindow *const resident_window =
        entries[index].recurrence.window;
    const std::uint32_t failed_outer_window =
        resident_window != nullptr && resident_window->nested()
            ? resident_window->outer_iteration
            : PreparedPipelineNoStep;
    const std::uint32_t failed_inner_iteration =
        resident_window != nullptr &&
                resident_window->phase == BackendWindowPhase::NestedAction
            ? resident_window->inner_iteration
            : PreparedPipelineNoStep;
    const std::uint32_t failed_nested_phase =
        resident_window == nullptr
            ? static_cast<std::uint32_t>(
                  rund::compute::PipelineNestedPhase::None)
            : static_cast<std::uint32_t>(resident_window->nested_phase());
    if (resident_window != nullptr &&
        resident_window->phase == BackendWindowPhase::NestedSeed &&
        !EncodeVulkanWindow(recording, pipeline->window,
                            static_cast<std::uint32_t>(index), true)) {
      return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
    }
    if (pipeline->profile != nullptr &&
        pipeline->profile->timestamps != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(recording, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          pipeline->profile->timestamps,
                          static_cast<std::uint32_t>(2u * index));
    }
    const PreparedProgramStatusSlice status_range =
        status_ranges[template_index];
    const PreparedProgramStatusSlice telemetry_range =
        telemetry_ranges[template_index];
    const std::size_t status_step_end =
        static_cast<std::size_t>(status_range.first) + status_range.count;
    const std::size_t telemetry_step_end =
        static_cast<std::size_t>(telemetry_range.first) + telemetry_range.count;
    if (status_range.count != resources->size() ||
        telemetry_range.count != resources->size() ||
        status_step_end > status_steps.size() ||
        telemetry_step_end > telemetry_steps.size()) {
      return FailVulkanPipeline(pipeline, "accel_kernel_run_invalid");
    }
    for (std::size_t step_index = 0u; step_index < resources->size();
         ++step_index) {
      VulkanKernelEntry *const step = resources->entry(step_index);
      if (step == nullptr) {
        return FailVulkanPipeline(pipeline, "accel_kernel_run_invalid");
      }
      rund::AccelCheck encoded{};
      if (resident_window == nullptr) {
        encoded = EncodeVulkanResets(*resources, step_index, recording);
        if (encoded.ok) {
          encoded = EncodeVulkanStep(*pipeline->adapter, *step, recording);
        }
      } else {
        VulkanDispatchScope scope{pipeline->window.capture, recording,
                                  resident_window->state};
        encoded = EncodeVulkanResets(*resources, step_index, recording);
        if (encoded.ok) {
          encoded = EncodeVulkanStep(*pipeline->adapter, *step, recording);
        }
      }
      if (!encoded.ok) {
        return FailVulkanPipeline(pipeline, encoded.reason);
      }

      const PreparedProgramStatusSlice status_slice =
          status_steps[status_range.first + step_index];
      const std::size_t status_end =
          static_cast<std::size_t>(status_slice.first) + status_slice.count;
      if (status_end > canonical.size()) {
        return FailVulkanPipeline(pipeline,
                                  "accel_kernel_primitive_unsupported");
      }
      if (status_slice.count != 0u) {
        EncodeVulkanComputeToComputeBarrier(recording);
      }
      for (std::size_t status_index = status_slice.first;
           status_index < status_end; ++status_index) {
        const VulkanPipelineCanonicalStatus &current = canonical[status_index];
        if (current.active_program != template_index) {
          return FailVulkanPipeline(pipeline,
                                    "accel_kernel_primitive_unsupported");
        }
        if (!EncodeVulkanPipelineCanonicalStatus(recording, pipeline->control,
                                                 current) ||
            !FoldVulkanPipelineControl(
                recording, pipeline->control,
                PreparedProgramStatusSlice{.first = current.first,
                                           .count = current.source.count},
                status.declared_steps[template_index], failed_outer_window,
                failed_inner_iteration, failed_nested_phase)) {
          return FailVulkanPipeline(pipeline,
                                    "accel_vulkan_command_unavailable");
        }
      }

      const PreparedProgramStatusSlice telemetry_slice =
          telemetry_steps[telemetry_range.first + step_index];
      const std::size_t telemetry_end =
          static_cast<std::size_t>(telemetry_slice.first) +
          telemetry_slice.count;
      if (telemetry_end > pipeline->telemetry.size() ||
          !EncodeVulkanTelemetry(
              *pipeline, recording,
              std::span<const VulkanPipelineTelemetryRecord>{
                  pipeline->telemetry}
                  .subspan(telemetry_slice.first, telemetry_slice.count),
              resident_window == nullptr
                  ? std::numeric_limits<std::uint32_t>::max()
                  : resident_window->state,
              status_slice.count == 0u)) {
        return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
      }
      if (status_slice.count != 0u && telemetry_slice.count == 0u) {
        EncodeVulkanComputeToComputeBarrier(recording);
      }
    }
    const PreparedProgramStatusSlice program_status =
        status.slices[template_index];
    std::uint64_t folded = 0u;
    for (std::size_t step_index = 0u; step_index < status_range.count;
         ++step_index) {
      const PreparedProgramStatusSlice slice =
          status_steps[status_range.first + step_index];
      if (slice.count != 0u) {
        const VulkanPipelineCanonicalStatus &current = canonical[slice.first];
        folded += current.source.count;
      }
    }
    if (folded != program_status.count) {
      return FailVulkanPipeline(pipeline, "accel_kernel_run_invalid");
    }
    if (!EncodeVulkanWindow(recording, pipeline->window,
                            static_cast<std::uint32_t>(index))) {
      return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
    }
    if (resident_window != nullptr &&
        (resident_window->phase == BackendWindowPhase::Ordinary ||
         resident_window->phase == BackendWindowPhase::NestedSeed) &&
        !EncodeVulkanPipelineSeal(recording, pipeline->publish,
                                  resident_window->state,
                                  resident_window->iteration)) {
      return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
    }
    if (pipeline->profile != nullptr &&
        pipeline->profile->timestamps != VK_NULL_HANDLE) {
      vkCmdWriteTimestamp(recording, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          pipeline->profile->timestamps,
                          static_cast<std::uint32_t>(2u * index + 1u));
      pipeline->profile->timestamped[index] = 1u;
    }
  }
  if (pipeline->window.capture.failed) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  if (!FreezeVulkanWindow(pipeline->window)) {
    return FailVulkanPipeline(pipeline, "accel_vulkan_memory_unavailable");
  }
  if (window_dispatches > pipeline->dispatch_count ||
      pipeline->window.capture.cursor >
          std::numeric_limits<std::uint64_t>::max() -
              (pipeline->dispatch_count - window_dispatches)) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  pipeline->dispatch_count =
      pipeline->dispatch_count - window_dispatches +
      static_cast<std::uint64_t>(pipeline->window.capture.cursor);
  if (pipeline->window.capture.indirect_count >
      std::numeric_limits<std::uint64_t>::max() - pipeline->dispatch_count) {
    return FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
  }
  pipeline->dispatch_count += pipeline->window.capture.indirect_count;
  if (!FinishVulkanPipelineControl(recording, pipeline->control, status)) {
    return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
  }
  if (!EncodeVulkanPipelinePublish(recording, pipeline->publish)) {
    return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
  }
  if (!PublishVulkanPipelineControl(recording, pipeline->control)) {
    return FailVulkanPipeline(pipeline, "accel_vulkan_command_unavailable");
  }
  const rund::AccelCheck ended = EndCommand(pipeline->command);
  if (!ended.ok) {
    return FailVulkanPipeline(pipeline, ended.reason);
  }
  std::uint64_t bytes =
      sizeof(VulkanPipeline) +
      (pipeline->profile == nullptr ? 0u : sizeof(VulkanPipelineProfile)) +
      pipeline->control.descriptor_leases.capacity() *
          sizeof(VulkanCollectiveDescriptorLease);
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, VulkanRecurrenceHostBytes(*pipeline));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, VulkanPipelinePublishHostBytes(pipeline->publish));
  bytes = ::rund::detail::counter::SaturatingAdd(
      bytes, VulkanWindowHostBytes(pipeline->window));
  memory.host = PreparedMemory{
      .current = bytes, .peak = bytes, .cumulative = bytes, .budget = bytes};
  prepared = std::static_pointer_cast<void>(pipeline);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
