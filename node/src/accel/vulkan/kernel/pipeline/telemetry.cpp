#include "telemetry.hpp"

#include "../../buffer/create/telemetry.hpp"
#include "../../collective/pipeline.hpp"
#include "../../command.hpp"
#include "../../descriptor.hpp"
#include "../../runtime/timestamp.hpp"
#include "source.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <new>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool PrepareVulkanTelemetry(VulkanPipeline &pipeline) {
  if (pipeline.telemetry.empty()) {
    return true;
  }
  const bool profile_steps = pipeline.profile != nullptr;
  rund::kernel::LoweringArtifact artifact{};
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.source_text =
      profile_steps ? VulkanProfileSource() : VulkanTelemetrySource();
  artifact.ok = true;
  artifact.reason = "ok";
  const std::uint32_t descriptor_count = profile_steps ? 5u : 4u;
  const std::uint32_t parameter_bytes =
      profile_steps ? sizeof(VulkanPipelineProfileTelemetryParams)
                    : sizeof(VulkanPipelineTelemetryParams);
  pipeline.telemetry_pipeline = AcquireVulkanCollectivePipeline(
      *pipeline.adapter, descriptor_count, parameter_bytes,
      profile_steps ? VulkanProfilePlan() : VulkanTelemetryPlan(), artifact);
  if (pipeline.telemetry_pipeline == nullptr ||
      pipeline.control.summary.buffer == VK_NULL_HANDLE ||
      (profile_steps && pipeline.control.profile.buffer == VK_NULL_HANDLE)) {
    return false;
  }
  auto *const prior = pipeline.adapter->active_descriptor_leases;
  pipeline.adapter->active_descriptor_leases =
      &pipeline.control.descriptor_leases;
  bool ready = true;
  try {
    for (VulkanPipelineTelemetryRecord &record : pipeline.telemetry) {
      const VulkanPipelineTelemetrySource &source = record.source;
      const VulkanBuffer *const primary = source.primary;
      const VulkanBuffer *const count =
          source.count == nullptr ? primary : source.count;
      const VulkanBuffer *const predicate =
          source.predicate == nullptr ? primary : source.predicate;
      const std::uint64_t primary_bytes =
          static_cast<std::uint64_t>(source.primary_word_count) *
          sizeof(std::uint32_t);
      if (primary == nullptr || count == nullptr || predicate == nullptr ||
          primary->buffer == VK_NULL_HANDLE || primary_bytes == 0u ||
          primary_bytes > primary->bytes ||
          source.count_offset / sizeof(std::uint32_t) >
              std::numeric_limits<std::uint32_t>::max() ||
          source.predicate_offset / sizeof(std::uint32_t) >
              std::numeric_limits<std::uint32_t>::max() ||
          !AcquireVulkanCollectiveDescriptorSet(
              *pipeline.adapter, *pipeline.telemetry_pipeline, descriptor_count,
              record.descriptor)) {
        ready = false;
        break;
      }
      if (profile_steps) {
        const std::array<const VulkanBuffer *, 5u> buffers{
            primary, count, predicate, &pipeline.control.summary,
            &pipeline.control.profile};
        ready = WriteVulkanStorageDescriptorSet(*pipeline.adapter,
                                                record.descriptor, buffers);
      } else {
        const std::array<const VulkanBuffer *, 4u> buffers{
            primary, count, predicate, &pipeline.control.summary};
        ready = WriteVulkanStorageDescriptorSet(*pipeline.adapter,
                                                record.descriptor, buffers);
      }
    }
  } catch (const std::bad_alloc &) {
    ready = false;
  }
  pipeline.adapter->active_descriptor_leases = prior;
  if (ready) {
    pipeline.control.command_count += pipeline.telemetry.size();
  }
  return ready;
}

[[nodiscard]] bool EncodeVulkanTelemetry(
    const VulkanPipeline &pipeline, const VkCommandBuffer command,
    const std::span<const VulkanPipelineTelemetryRecord> telemetry,
    const bool visible) noexcept {
  if (telemetry.empty()) {
    return true;
  }
  if (command == VK_NULL_HANDLE || pipeline.telemetry_pipeline == nullptr) {
    return false;
  }
  if (visible) {
    EncodeVulkanComputeToComputeBarrier(command);
  }
  for (const VulkanPipelineTelemetryRecord &record : telemetry) {
    const VulkanPipelineTelemetrySource &source = record.source;
    if (record.descriptor == VK_NULL_HANDLE) {
      return false;
    }
    const VulkanPipelineTelemetryParams params{
        .kind = static_cast<std::uint32_t>(source.kind),
        .primary_word_count = source.primary_word_count,
        .count_source = static_cast<std::uint32_t>(source.control.count_source),
        .predicate_source =
            static_cast<std::uint32_t>(source.control.predicate_source),
        .has_count = source.control.has_count() ? 1u : 0u,
        .has_predicate = source.control.has_predicate() ? 1u : 0u,
        .iteration = source.control.iteration,
        .count_word_offset = static_cast<std::uint32_t>(source.count_offset /
                                                        sizeof(std::uint32_t)),
        .predicate_word_offset = static_cast<std::uint32_t>(
            source.predicate_offset / sizeof(std::uint32_t)),
        .indirect_dispatch_count = source.indirect_dispatch_count,
        .declared_step = record.declared_step,
        .capacity = source.capacity,
        .predicate_expected = source.control.predicate_expected,
        .work_item_count = source.work_item_count,
    };
    BindVulkanPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                       pipeline.telemetry_pipeline->pipeline);
    BindVulkanDescriptors(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline.telemetry_pipeline->pipeline_layout, 0u, 1u,
                          &record.descriptor, 0u, nullptr);
    if (pipeline.profile != nullptr) {
      const VulkanPipelineProfileTelemetryParams profile_params{
          .telemetry = params,
          .declared_step_count = pipeline.profile->declared_step_count,
      };
      PushVulkanConstants(command, pipeline.telemetry_pipeline->pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                          sizeof(profile_params), &profile_params);
    } else {
      PushVulkanConstants(command, pipeline.telemetry_pipeline->pipeline_layout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(params),
                          &params);
    }
    DispatchVulkan(command, 1u, 1u, 1u);
    EncodeVulkanComputeToComputeBarrier(command);
  }
  return true;
}

[[nodiscard]] std::uint64_t timestamp_ns(const VulkanAdapter &adapter,
                                         const std::uint64_t start,
                                         const std::uint64_t end) noexcept {
  const std::uint64_t ticks =
      VulkanTimestampTicks(start, end, adapter.timestamp_valid_bits);
  const long double elapsed_ns =
      static_cast<long double>(ticks) *
      static_cast<long double>(adapter.timestamp_period_ns);
  const long double limit =
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
  return elapsed_ns >= limit
             ? std::numeric_limits<std::uint64_t>::max()
             : static_cast<std::uint64_t>(std::max(0.0L, elapsed_ns));
}

void ObserveVulkanProfile(VulkanPipeline &pipeline,
                          KernelResult &result) noexcept {
  if (pipeline.profile == nullptr) {
    return;
  }
  VulkanPipelineProfile &profile = *pipeline.profile;
  result.pipeline.profile = PreparedPipelineProfileEvidence{
      .instrumentation_command_count = profile.instrumentation_command_count,
      .instrumentation_byte_count = profile.instrumentation_byte_count,
  };
  if (!ReadVulkanPipelineProfile(
          pipeline.control,
          std::span<PreparedPipelineStepControl>{
              profile.controls.data(), profile.declared_step_count})) {
    return;
  }
  for (std::size_t declared = 0u; declared < profile.declared_step_count;
       ++declared) {
    PreparedPipelineStepEvidence &row = profile.rows[declared];
    const bool visible =
        result.pipeline.control.reason == 0u ||
        (result.pipeline.control.failed_step != PreparedPipelineNoStep &&
         declared <= result.pipeline.control.failed_step);
    row.control =
        visible ? profile.controls[declared] : PreparedPipelineStepControl{};
    row.duration_ns = 0u;
    row.timing_sample_count = 0u;
    row.work_sample_count = visible ? 1u : 0u;
    row.clock = PreparedPipelineStepClock::Unavailable;
    row.relation = PreparedPipelineStepTimingRelation::Unavailable;
  }
  if (profile.timestamps != VK_NULL_HANDLE && profile.query_count != 0u) {
    std::array<std::uint64_t, PreparedPipelineStepCapacity * 2u> timestamps{};
    const VkResult queried = vkGetQueryPoolResults(
        pipeline.adapter->device, profile.timestamps, 0u, profile.query_count,
        static_cast<std::size_t>(profile.query_count) * sizeof(std::uint64_t),
        timestamps.data(), sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (queried == VK_SUCCESS) {
      for (std::size_t active = 0u; active < profile.active_step_count;
           ++active) {
        if (!profile.timestamped[active]) {
          continue;
        }
        const std::uint32_t declared = profile.declared_steps[active];
        if (declared >= profile.declared_step_count) {
          continue;
        }
        PreparedPipelineStepEvidence &row = profile.rows[declared];
        if (row.work_sample_count == 0u) {
          continue;
        }
        row.duration_ns =
            timestamp_ns(*pipeline.adapter, timestamps[2u * active],
                         timestamps[2u * active + 1u]);
        row.timing_sample_count = 1u;
        row.clock = PreparedPipelineStepClock::Device;
        row.relation = PreparedPipelineStepTimingRelation::NonAdditive;
      }
    }
  }
  result.pipeline.profile = PreparedPipelineProfileEvidence{
      .steps =
          std::span<const PreparedPipelineStepEvidence>{
              profile.rows.data(), profile.declared_step_count},
      .instrumentation_command_count = profile.instrumentation_command_count,
      .instrumentation_byte_count = profile.instrumentation_byte_count,
      .observed = true,
  };
}

#endif

} // namespace rund::node::accel::detail
