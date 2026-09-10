#include "internal.hpp"

#include "../../state.hpp"

namespace rund::node::accel::detail::metal_pipeline_admit_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck ValidateMetalPipelineInput(MetalPipelineBuild &build,
                                            bool &compact_input) {
  if (build.templates.empty() ||
      build.templates.size() != build.status.active_step_count ||
      build.templates.front().run == nullptr ||
      build.templates.front().run->pick == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  compact_input = build.entries.empty() && build.barriers.empty() &&
                  build.status.command_count == 2u &&
                  build.aggregates.size() == 1u;
  if (!compact_input) {
    if (build.entries.empty() ||
        build.entries.size() != build.barriers.size() ||
        build.entries.size() != build.status.command_count ||
        build.entries.front().run == nullptr ||
        build.entries.front().run->pick == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    for (std::size_t index = 0u; index < build.entries.size(); ++index) {
      build.failure_context.occurrence_route(build.entries[index]);
      if (build.entries[index].occurrence_index != index ||
          build.entries[index].template_index >= build.templates.size() ||
          (build.entries[index].transducer != NoTileTransducer &&
           build.entries[index].transducer >= build.transducers.size())) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
    }
  }
  build.failure_context.clear_route();
  const rund::AccelCheck valid = ValidateMetalKernelContext(
      *build.templates.front().run->pick, build.context);
  if (!valid.ok || build.context.adapter == nullptr) {
    return valid;
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_admit_internal
