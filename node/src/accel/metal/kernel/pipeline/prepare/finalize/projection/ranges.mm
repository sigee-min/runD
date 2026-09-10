#include "../internal.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck
FinalizeMetalProjectionRanges(MetalPipelineBuild &build,
                              MetalPipelineFinalizeProjection &projection) {
  const std::uint32_t no_declared_step =
      std::numeric_limits<std::uint32_t>::max();
  std::size_t first_step_command = build.captured.commands.size();
  std::size_t last_step_command = 0u;
  bool selectable =
      !build.pipeline->residency_steps.empty() && !build.recurrence.ready() &&
      !build.aggregate_selected && !build.profile_steps &&
      (build.native_windows.empty() ? build.native_publication_count == 0u
                                    : projection.direct_window);
  for (const BackendBatchEntry &entry : build.entries) {
    if (entry.template_index >= build.status.active_step_count ||
        build.status.declared_steps[entry.template_index] >=
            build.pipeline->residency_steps.size()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const auto *const resources =
        static_cast<const MetalKernelResources *>(entry.prepared->get());
    if (resources == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    MetalResidencyStepRange &range =
        build.pipeline->residency_steps
            [build.status.declared_steps[entry.template_index]];
    range.reset_count = ::rund::detail::counter::SaturatingAdd(
        range.reset_count, resources->reset_count);
    range.reset_bytes = ::rund::detail::counter::SaturatingAdd(
        range.reset_bytes, resources->reset_bytes);
  }
  for (std::size_t index = 0u; index < build.captured.commands.size();
       ++index) {
    const MetalCommand &command = build.captured.commands[index];
    if (command.declared_step == no_declared_step) {
      continue;
    }
    if (command.declared_step >= build.pipeline->residency_steps.size()) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    MetalResidencyStepRange &range =
        build.pipeline->residency_steps[command.declared_step];
    if (range.count == 0u) {
      range.begin = index;
    } else if (range.begin + range.count != index) {
      selectable = false;
    }
    ++range.count;
    range.dispatch_count += static_cast<std::uint64_t>(command.trace);
    range.control_count += static_cast<std::uint64_t>(command.control);
    first_step_command = std::min(first_step_command, index);
    last_step_command = index + 1u;
  }
  if (first_step_command == build.captured.commands.size() ||
      last_step_command <= first_step_command) {
    selectable = false;
  } else {
    for (std::size_t index = first_step_command; index < last_step_command;
         ++index) {
      selectable = selectable && build.captured.commands[index].declared_step !=
                                     no_declared_step;
    }
  }
  for (const MetalResidencyStepRange &range : build.pipeline->residency_steps) {
    selectable = selectable && range.valid() && range.dispatch_count != 0u;
  }
  bool spatial_window = projection.spatial_window;
  if (spatial_window) {
    for (std::size_t local = 0u;
         local < build.pipeline->spatial_window.local_count; ++local) {
      const MetalResidencyStepRange &range =
          build.pipeline->residency_steps[local];
      MetalSpatialWindowLocalProof &proof =
          build.pipeline->spatial_window.locals[local];
      proof.range_begin = range.begin;
      proof.range_count = range.count;
      proof.range_dispatch_count = range.dispatch_count;
      spatial_window = spatial_window && proof.valid() &&
                       build.pipeline->spatial_window.local_matches(local);
    }
  }
  projection.spatial_window = spatial_window;
  projection.native_window_capacity = build.native_windows.capacity();
  build.pipeline->residency_prefix = MetalResidencyStepRange{
      .begin = 0u,
      .count = first_step_command,
      .dispatch_count = 0u,
      .control_count = 0u,
  };
  build.pipeline->residency_suffix = MetalResidencyStepRange{
      .begin = last_step_command,
      .count = build.captured.commands.size() - last_step_command,
      .dispatch_count = 0u,
      .control_count = 0u,
  };
  const auto trace_count = [&](const std::size_t begin,
                               const std::size_t end) noexcept {
    std::uint64_t count = 0u;
    for (std::size_t index = begin; index < end; ++index) {
      count += static_cast<std::uint64_t>(build.captured.commands[index].trace);
    }
    return count;
  };
  const auto control_count = [&](const std::size_t begin,
                                 const std::size_t end) noexcept {
    std::uint64_t count = 0u;
    for (std::size_t index = begin; index < end; ++index) {
      count +=
          static_cast<std::uint64_t>(build.captured.commands[index].control);
    }
    return count;
  };
  build.pipeline->residency_prefix.dispatch_count =
      trace_count(0u, first_step_command);
  build.pipeline->residency_prefix.control_count =
      control_count(0u, first_step_command);
  build.pipeline->residency_suffix.dispatch_count =
      trace_count(last_step_command, build.captured.commands.size());
  build.pipeline->residency_suffix.control_count =
      control_count(last_step_command, build.captured.commands.size());
  build.pipeline->residency_selectable =
      selectable && build.pipeline->residency_prefix.dispatch_count == 0u &&
      build.pipeline->residency_suffix.dispatch_count == 0u;
  build.pipeline->persistent_spatial_window_selectable =
      spatial_window && selectable &&
      build.pipeline->residency_prefix.dispatch_count == 0u &&
      build.pipeline->residency_suffix.dispatch_count == 0u;
  if (build.pipeline->persistent_spatial_window_selectable &&
      !build.pipeline->spatial_window_proof_valid()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  // A fused top-level recurrence is an aggregate-only native route. It can
  // never be submitted through the per-declared-step Residency selection
  // API, so retaining one unused MetalResidencyStepRange per authored
  // iteration would make the live prepared owner grow by 48 bytes per Q while
  // the executable command stream is fixed. Release that cold projection
  // before retained-memory publication; fallback Pipelines keep it unchanged.
  if (build.recurrence.ready()) {
    std::vector<MetalResidencyStepRange>{}.swap(
        build.pipeline->residency_steps);
    build.pipeline->residency_prefix = {};
    build.pipeline->residency_suffix = {};
    build.pipeline->residency_selectable = false;
  }
  if (projection.uses_parameters && build.captured.parameters.empty()) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
