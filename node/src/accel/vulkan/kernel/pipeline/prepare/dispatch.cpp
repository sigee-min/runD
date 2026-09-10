#include "context.hpp"
#include <limits>
#include <rund/counter.hpp>
namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanPipelineOccurrences(VulkanPipelinePreparation &preparation) {
  auto &pipeline = preparation.pipeline;
  auto &status_steps = preparation.status_steps;
  auto &telemetry_steps = preparation.telemetry_steps;
  auto &status_ranges = preparation.status_ranges;
  auto &telemetry_ranges = preparation.telemetry_ranges;
  auto &template_work = preparation.template_work;
  auto &transducer_work = preparation.transducer_work;
  auto &transducer_occurrences = preparation.transducer_occurrences;
  auto &window_dispatches = preparation.window_dispatches;
  auto &window_gate_count = preparation.window_gate_count;
  auto &status_command_sources = preparation.status_command_sources;
  auto &telemetry_command_count = preparation.telemetry_command_count;
  auto &encoded_work_command_count = preparation.encoded_work_command_count;
  const auto &templates = preparation.templates;
  const auto &entries = preparation.entries;
  const auto &transducers = preparation.transducers;
  auto &status = *preparation.status;
  auto &failure_context = *preparation.failure_context;
  const auto &recurrence = preparation.recurrence;
  failure_context.stage(PreparedPipelineFailureStage::BackendDescription);
  for (std::size_t entry_index = 0u;
       !recurrence.ready() && entry_index < entries.size(); ++entry_index) {
    const BackendBatchEntry &entry = entries[entry_index];
    failure_context.occurrence_route(entry);
    if (entry.template_index >= templates.size() ||
        entry.occurrence_index != entry_index) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<VulkanKernelResources *>(entry.prepared->get());
    if (resources == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const TileTransducer *const transducer =
        entry.transducer == NoTileTransducer ? nullptr
                                             : &transducers[entry.transducer];
    const BackendWindow *const window = entry.recurrence.window;
    if (transducer != nullptr &&
        (!transducer->recurrence.ready() || window == nullptr ||
         !window->valid_occurrence(true) ||
         transducer->template_first != entry.template_index ||
         transducer->template_count != window->inner_bound)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    std::uint64_t captured_dispatches = 0u;
    std::uint64_t captured_indirect = 0u;
    const rund::AccelCheck described =
        transducer == nullptr
            ? DescribeVulkanRouteDispatches(*resources, captured_dispatches,
                                            captured_indirect)
            : rund::AccelCheck{true, "ok"};
    if (!described.ok) {
      return described;
    }
    // Public dispatch evidence owns algorithmic Program work.  Vulkan's
    // frozen resident-window arena instead owns every raw vkCmdDispatch and
    // vkCmdDispatchIndirect that must be rewritten.  Multi-stage primitives
    // such as Sort therefore need the exact captured count for the arena, but
    // must retain the authored Program count in Stats and profile evidence.
    const std::uint64_t physical_dispatches =
        transducer == nullptr ? resources->dispatch_count
                              : transducer->recurrence.window_count;
    // A compact Action transducer has no ordinary Program manifest, but its
    // recurrence still emits one captured direct dispatch per frozen map
    // window.  Charge those slots to the resident-window arena exactly as the
    // later EncodeVulkanMap call consumes them.  Treating the transducer as
    // zero captured work under-reserved by one slot per outer occurrence and
    // rejected a valid final Seed/Fold route at capture time.
    if (transducer != nullptr) {
      captured_dispatches = physical_dispatches;
    }
    const std::uint64_t encoded_work =
        transducer == nullptr ? captured_dispatches : physical_dispatches;
    if (physical_dispatches > std::numeric_limits<std::uint64_t>::max() -
                                  pipeline->dispatch_count ||
        !rund::kernel::checked::add(encoded_work_command_count, encoded_work,
                                    encoded_work_command_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    pipeline->dispatch_count += physical_dispatches;
    if (window != nullptr) {
      if (captured_dispatches >
              std::numeric_limits<std::uint64_t>::max() - window_dispatches ||
          captured_indirect >
              std::numeric_limits<std::uint64_t>::max() - window_gate_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      window_dispatches += captured_dispatches;
      window_gate_count += captured_indirect;
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
          row.final_dispatch_count, transducer == nullptr
                                        ? entry.run->final_dispatch_count
                                        : physical_dispatches);
      row.physical_dispatch_count = ::rund::detail::counter::SaturatingAdd(
          row.physical_dispatch_count, physical_dispatches);
      const VulkanPipelineWork &work = transducer == nullptr
                                           ? template_work[entry.template_index]
                                           : transducer_work[entry.transducer];
      if (work.exact && work.dispatch_count == physical_dispatches) {
        row.workgroup_count = ::rund::detail::counter::SaturatingAdd(
            row.workgroup_count, work.workgroup_count);
        row.work_item_count = ::rund::detail::counter::SaturatingAdd(
            row.work_item_count, work.work_item_count);
      }
      if (transducer != nullptr) {
        transducer_occurrences[entry.transducer] =
            ::rund::detail::counter::SaturatingAdd(
                transducer_occurrences[entry.transducer], 1u);
      }
    }
    const PreparedProgramStatusSlice status_range =
        status_ranges[entry.template_index];
    const PreparedProgramStatusSlice telemetry_range =
        telemetry_ranges[entry.template_index];
    const std::size_t status_end =
        static_cast<std::size_t>(status_range.first) + status_range.count;
    const std::size_t telemetry_end =
        static_cast<std::size_t>(telemetry_range.first) + telemetry_range.count;
    if (status_end > status_steps.size() ||
        telemetry_end > telemetry_steps.size() ||
        (transducer != nullptr &&
         (resources->size() != 1u || status_range.count != 1u ||
          telemetry_range.count != 1u ||
          status.slices[entry.template_index].count != 0u ||
          status_steps[status_range.first].count != 0u ||
          telemetry_steps[telemetry_range.first].count != 0u))) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
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
  if (pipeline->profile != nullptr) {
    for (std::size_t index = 0u; index < transducers.size(); ++index) {
      const TileTransducer &transducer = transducers[index];
      for (std::uint32_t offset = 1u; offset < transducer.template_count;
           ++offset) {
        const std::size_t template_index = transducer.template_first + offset;
        failure_context.template_route(
            static_cast<std::uint32_t>(template_index));
        const BackendBatchEntry &entry = templates[template_index];
        const std::uint32_t declared = status.declared_steps[template_index];
        if (entry.run == nullptr || declared >= status.declared_step_count) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
        const std::uint64_t original =
            entry.run->original_dispatch_count != 0u &&
                    transducer_occurrences[index] >
                        std::numeric_limits<std::uint64_t>::max() /
                            entry.run->original_dispatch_count
                ? std::numeric_limits<std::uint64_t>::max()
                : transducer_occurrences[index] *
                      entry.run->original_dispatch_count;
        PreparedPipelineStepEvidence &row = pipeline->profile->rows[declared];
        row.original_dispatch_count = ::rund::detail::counter::SaturatingAdd(
            row.original_dispatch_count, original);
      }
    }
  }


  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
