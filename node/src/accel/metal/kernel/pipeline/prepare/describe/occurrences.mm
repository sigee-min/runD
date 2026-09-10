#include "internal.hpp"

#include <limits>

#include <rund/counter.hpp>

namespace rund::node::accel::detail::metal_pipeline_describe_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck DescribeMetalOccurrences(MetalPipelineBuild &build) {
  // Dispatch/reset evidence follows physical command occurrences. Profile
  // rows aggregate repeated occurrences back into their declared compact step.
  for (std::size_t entry_index = 0u; entry_index < build.entries.size();
       ++entry_index) {
    const BackendBatchEntry &entry = build.entries[entry_index];
    build.failure_context.occurrence_route(entry);
    if (entry.template_index >= build.templates.size() ||
        entry.occurrence_index != entry_index ||
        entry.run != build.templates[entry.template_index].run ||
        entry.prepared != build.templates[entry.template_index].prepared) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    auto *const resources =
        entry.prepared == nullptr
            ? nullptr
            : static_cast<MetalKernelResources *>(entry.prepared->get());
    if (resources == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const TileTransducer *const transducer =
        entry.transducer == NoTileTransducer
            ? nullptr
            : &build.transducers[entry.transducer];
    const std::uint64_t physical_dispatches =
        transducer == nullptr ? resources->dispatch_count
                              : transducer->recurrence.window_count;
    const std::uint32_t declared_step =
        build.status.declared_steps[entry.template_index];
    if (declared_step >= build.status.declared_step_count) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    if (build.recurrence.ready()) {
      if (entry_index == 0u) {
        build.pipeline->dispatch_count = build.recurrence.window_count;
      }
    } else {
      if (physical_dispatches > std::numeric_limits<std::uint64_t>::max() -
                                    build.pipeline->dispatch_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      build.pipeline->dispatch_count += physical_dispatches;
      build.pipeline->reset_count = ::rund::detail::counter::SaturatingAdd(
          build.pipeline->reset_count, resources->reset_count);
      build.pipeline->reset_bytes = ::rund::detail::counter::SaturatingAdd(
          build.pipeline->reset_bytes, resources->reset_bytes);
    }
    if (build.profile_steps) {
      const bool recurrence_owner =
          build.recurrence.ready() && entry_index == 0u;
      PreparedPipelineStepEvidence &row =
          build.pipeline->step_evidence[declared_step];
      const std::uint64_t original = entry.run->original_dispatch_count;
      const std::uint64_t final =
          build.recurrence.ready()
              ? (recurrence_owner ? build.recurrence.window_count : 0u)
              : (transducer == nullptr ? entry.run->final_dispatch_count
                                       : physical_dispatches);
      const std::uint64_t physical =
          build.recurrence.ready()
              ? (recurrence_owner ? build.recurrence.window_count : 0u)
              : physical_dispatches;
      if (original > std::numeric_limits<std::uint64_t>::max() -
                         row.original_dispatch_count ||
          final > std::numeric_limits<std::uint64_t>::max() -
                      row.final_dispatch_count ||
          physical > std::numeric_limits<std::uint64_t>::max() -
                         row.physical_dispatch_count) {
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      row.original_dispatch_count += original;
      row.final_dispatch_count += final;
      row.physical_dispatch_count += physical;
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_describe_internal
