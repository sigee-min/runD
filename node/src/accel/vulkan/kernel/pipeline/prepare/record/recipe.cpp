#include "../record.hpp"

#include "../../../../../kernel/backend/exception.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

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
    if (!recurrence) {
      recipe.entries.reserve(entries.size());
      for (const BackendBatchEntry &entry : entries) {
        if (entry.run == nullptr || entry.prepared == nullptr ||
            *entry.prepared == nullptr) {
          return rund::AccelCheck{false, "accel_kernel_run_invalid"};
        }
        recipe.entries.push_back(VulkanPipelineRecordEntry{
            .run = entry.run,
            .prepared = *entry.prepared,
            .window =
                entry.recurrence.window == nullptr
                    ? std::optional<BackendWindow>{}
                    : std::optional<BackendWindow>{*entry.recurrence.window},
            .transducer = entry.transducer,
            .template_index = entry.template_index,
            .occurrence_index = entry.occurrence_index,
            .step_index = 0u,
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
    }
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
  const bool exact = recurrence
                         ? recipe.entries.empty() && recipe.barriers.empty()
                         : recipe.entries.size() == entries.size() &&
                               recipe.barriers.size() == barriers.size();
  return exact ? rund::AccelCheck{true, "ok"}
               : rund::AccelCheck{false, "compute_pipeline_capacity"};
}

#endif

} // namespace rund::node::accel::detail
