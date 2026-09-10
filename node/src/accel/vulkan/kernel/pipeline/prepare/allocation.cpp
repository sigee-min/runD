#include "context.hpp"
#include <memory>
#include <limits>
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanPipelineAllocation(VulkanPipelinePreparation &preparation) {
  auto &pipeline = preparation.pipeline;
  auto &canonical = preparation.canonical;
  auto &status_steps = preparation.status_steps;
  auto &telemetry_steps = preparation.telemetry_steps;
  auto &recurrence_staging = preparation.recurrence_staging;
  auto &transducer_work = preparation.transducer_work;
  auto &transducer_occurrences = preparation.transducer_occurrences;
  auto &encoded_work_command_count = preparation.encoded_work_command_count;
  auto &description_capacity = preparation.description_capacity;
  const bool profile_steps = preparation.profile_steps;
  auto *const memory_meter = preparation.memory_meter;
  const auto &entries = preparation.entries;
  const auto &transducers = preparation.transducers;
  auto &registry = *preparation.registry;
  auto &status = *preparation.status;
  auto &failure_context = *preparation.failure_context;
  const auto &recurrence = preparation.recurrence;
  failure_context.stage(PreparedPipelineFailureStage::BackendAllocation);
  {
    pipeline = std::make_shared<VulkanPipeline>();
    pipeline->memory_meter = memory_meter;
    canonical.reserve(
        static_cast<std::size_t>(description_capacity.status_source_count));
    status_steps.reserve(
        static_cast<std::size_t>(description_capacity.step_count));
    telemetry_steps.reserve(
        static_cast<std::size_t>(description_capacity.step_count));
    pipeline->telemetry.reserve(
        static_cast<std::size_t>(description_capacity.telemetry_source_count));
    if (profile_steps) {
      pipeline->profile = std::make_unique<VulkanPipelineProfile>();
      transducer_work.resize(transducers.size());
      transducer_occurrences.resize(transducers.size());
    }
  }
  failure_context.stage(PreparedPipelineFailureStage::BackendAdmission);
  if (pipeline->profile != nullptr) {
    if (status.active_step_count > PreparedPipelineStepCapacity ||
        status.declared_step_count > PreparedPipelineStepCapacity ||
        status.declared_step_count == 0u) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    pipeline->profile->declared_step_count = status.declared_step_count;
    pipeline->profile->declared_steps = status.declared_steps;
    std::array<bool, PreparedPipelineStepCapacity> declared{};
    for (std::size_t active = 0u; active < status.active_step_count; ++active) {
      failure_context.template_route(static_cast<std::uint32_t>(active));
      const std::uint32_t ordinal = status.declared_steps[active];
      if (ordinal >= status.declared_step_count || declared[ordinal]) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      declared[ordinal] = true;
    }
  }
  if (recurrence.ready()) {
    failure_context.stage(PreparedPipelineFailureStage::BackendAllocation);
    const rund::AccelCheck ready = PrepareVulkanRecurrence(
        entries, recurrence, registry, status, *pipeline, recurrence_staging);
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
        failure_context.occurrence_route(entry);
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
    encoded_work_command_count = pipeline->dispatch_count;
  }

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
