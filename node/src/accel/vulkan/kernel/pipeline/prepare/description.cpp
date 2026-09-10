#include "context.hpp"
#include <limits>
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanPipelineDescriptions(VulkanPipelinePreparation &preparation) {
  auto &pipeline = preparation.pipeline;
  auto &canonical = preparation.canonical;
  auto &status_steps = preparation.status_steps;
  auto &telemetry_steps = preparation.telemetry_steps;
  auto &status_ranges = preparation.status_ranges;
  auto &telemetry_ranges = preparation.telemetry_ranges;
  auto &template_work = preparation.template_work;
  auto &transducer_staging = preparation.transducer_staging;
  auto &transducer_work = preparation.transducer_work;
  auto &described_status_entry_count = preparation.described_status_entry_count;
  auto &description_capacity = preparation.description_capacity;
  const auto &templates = preparation.templates;
  const auto &transducers = preparation.transducers;
  auto &registry = *preparation.registry;
  auto &status = *preparation.status;
  auto &failure_context = *preparation.failure_context;
  const auto &recurrence = preparation.recurrence;
  failure_context.stage(PreparedPipelineFailureStage::BackendDescription);
  // Describe immutable status and telemetry once per compact route template.
  // Expanded physical occurrences reuse these slices during native capture.
  for (std::size_t template_index = 0u;
       !recurrence.ready() && template_index < templates.size();
       ++template_index) {
    const BackendBatchEntry &entry = templates[template_index];
    failure_context.template_route(static_cast<std::uint32_t>(template_index));
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
    {
      for (std::size_t step_index = 0u; step_index < resources->size();
           ++step_index) {
        failure_context.template_node_route(entry, step_index);
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
              .source_node =
                  entry.run == nullptr || entry.run->steps == nullptr ||
                          step_index >= entry.run->step_count ||
                          entry.run->steps[step_index].step == nullptr
                      ? PreparedPipelineNoStep
                      : entry.run->steps[step_index].step->source.begin.index,
          });
          status_slice.count = 1u;
          entry_count += source.count;
        }
        status_steps.push_back(status_slice);
      }
      failure_context.template_route(
          static_cast<std::uint32_t>(template_index));
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
              entry_count) ||
          !rund::kernel::checked::add(described_status_entry_count, entry_count,
                                      described_status_entry_count)) {
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
    }
  }
  if (!recurrence.ready() &&
      (canonical.size() != description_capacity.status_source_count ||
       described_status_entry_count !=
           description_capacity.status_entry_count ||
       status_steps.size() != description_capacity.step_count ||
       telemetry_steps.size() != description_capacity.step_count ||
       pipeline->telemetry.size() !=
           description_capacity.telemetry_source_count)) {
    return rund::AccelCheck{false, "compute_dispatch_count_mismatch"};
  }
  if (!recurrence.ready()) {
    failure_context.stage(PreparedPipelineFailureStage::BackendAllocation);
    failure_context.clear_route();
    const rund::AccelCheck ready =
        PrepareVulkanTransducers(templates, transducers, registry, status,
                                 *pipeline, transducer_staging);
    if (!ready.ok) {
      return ready;
    }
    if (pipeline->transducers.size() != transducers.size()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    if (pipeline->profile != nullptr) {
      for (std::size_t index = 0u; index < transducers.size(); ++index) {
        failure_context.template_route(transducers[index].template_first);
        const auto *const map = static_cast<const VulkanMapEncodeResources *>(
            pipeline->transducers[index].get());
        VulkanPipelineWork &work = transducer_work[index];
        if (map == nullptr || !AccumulateVulkanWork(*map, work) ||
            work.dispatch_count != transducers[index].recurrence.window_count) {
          work = VulkanPipelineWork{.exact = false};
        }
      }
    }
  }

  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
