#include "internal.hpp"

#include "../../../../runtime/map/api.hpp"

#include <limits>

namespace rund::node::accel::detail::metal_pipeline_program_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck EncodeRecurrence(MetalPipelineBuild &build) {
  if (build.entries.empty() || !build.status_bindings.empty() ||
      !build.pipeline->telemetry.empty() ||
      build.pipeline->recurrence == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  build.failure_context.occurrence_route(build.entries.front());
  build.captured.declared_step = build.status.declared_steps[0u];
  const rund::AccelCheck encoded =
      EncodeMetalMap(*build.pipeline->adapter, build.pipeline->recurrence,
                     (__bridge void *)build.encoder);
  if (!encoded.ok) {
    return encoded;
  }
  const rund::AccelCheck capture = CheckMetalPipelineCapture(build.captured);
  if (!capture.ok) {
    return capture;
  }
  for (std::size_t command_index = build.reset_command_count;
       command_index < build.captured.commands.size(); ++command_index) {
    build.captured.commands[command_index].trace = true;
  }
  if (build.profile_steps) {
    const MetalWork work = MeasureMetalWork(std::span<const MetalCommand>{
        build.captured.commands.data() + build.reset_command_count,
        build.captured.commands.size() - build.reset_command_count});
    if (work.exact) {
      PreparedPipelineStepEvidence &row =
          build.pipeline->step_evidence[build.status.declared_steps[0u]];
      row.workgroup_count = work.workgroup_count;
      row.work_item_count = work.work_item_count;
    }
  }
  build.captured.declared_step = std::numeric_limits<std::uint32_t>::max();
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_program_internal
