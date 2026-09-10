#include "internal.hpp"

#include <limits>

#include <rund/counter.hpp>

namespace rund::node::accel::detail::metal_pipeline_describe_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck DescribeMetalProfile(MetalPipelineBuild &build) {
  if (!build.profile_steps) {
    return rund::AccelCheck{true, "ok"};
  }
  if (build.transducers.size() > PreparedPipelineStepCapacity) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  // Transducer identity is already bounded by the frozen compact route
  // table. Profiling only needs one counter per identity, so a heap mirror
  // would add a cold owner without adding information.
  std::array<std::uint64_t, PreparedPipelineStepCapacity> occurrence_counts{};
  for (const BackendBatchEntry &entry : build.entries) {
    if (entry.transducer != NoTileTransducer) {
      occurrence_counts[entry.transducer] =
          ::rund::detail::counter::SaturatingAdd(
              occurrence_counts[entry.transducer], 1u);
    }
  }
  for (std::size_t index = 0u; index < build.transducers.size(); ++index) {
    const TileTransducer &transducer = build.transducers[index];
    for (std::uint32_t offset = 1u; offset < transducer.template_count;
         ++offset) {
      const std::size_t template_index = transducer.template_first + offset;
      build.failure_context.template_route(
          static_cast<std::uint32_t>(template_index));
      const BackendBatchEntry &entry = build.templates[template_index];
      const std::uint32_t declared =
          build.status.declared_steps[template_index];
      if (entry.run == nullptr ||
          declared >= build.status.declared_step_count) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      const std::uint64_t original =
          entry.run->original_dispatch_count != 0u &&
                  occurrence_counts[index] >
                      std::numeric_limits<std::uint64_t>::max() /
                          entry.run->original_dispatch_count
              ? std::numeric_limits<std::uint64_t>::max()
              : occurrence_counts[index] * entry.run->original_dispatch_count;
      PreparedPipelineStepEvidence &row =
          build.pipeline->step_evidence[declared];
      row.original_dispatch_count = ::rund::detail::counter::SaturatingAdd(
          row.original_dispatch_count, original);
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_describe_internal
