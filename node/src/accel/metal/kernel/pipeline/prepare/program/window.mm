#include "internal.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::node::accel::detail::metal_pipeline_program_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck EncodeWindowControl(MetalPipelineBuild &build,
                                     const std::size_t entry_index,
                                     const BackendWindow *const resident_window,
                                     const std::uint32_t stage) {
  const auto window_begin = std::lower_bound(
      build.native_windows.begin(), build.native_windows.end(), entry_index,
      [](const MetalWindow &window, const std::size_t entry) {
        return window.entry < entry;
      });
  const auto window_end =
      std::upper_bound(window_begin, build.native_windows.end(), entry_index,
                       [](const std::size_t entry, const MetalWindow &window) {
                         return entry < window.entry;
                       });
  bool encoded = false;
  for (auto route = window_begin; route != window_end; ++route) {
    BackendWindowPhase phase{};
    if (!DecodeBackendWindowPhase(route->params.phase, phase)) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const bool selected =
        phase == BackendWindowPhase::NestedSeed
            ? stage == 0u
            : (phase == BackendWindowPhase::NestedAction
                   ? stage == 2u && route->params.inner_advance != 0u
                   : (phase == BackendWindowPhase::NestedFold
                          ? stage == 2u && resident_window != nullptr &&
                                resident_window->outer_iteration + 1u ==
                                    resident_window->outer_bound
                          : stage == 1u));
    if (!selected) {
      continue;
    }
    if (!encoded) {
      [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      encoded = true;
    }
    id<MTLBuffer> const resident =
        (__bridge id<MTLBuffer>)route->resident.get();
    std::array<id<MTLBuffer>, 3u> terminals{
        (__bridge id<MTLBuffer>)route->terminals[0].get(),
        (__bridge id<MTLBuffer>)route->terminals[1].get(),
        (__bridge id<MTLBuffer>)route->terminals[2].get()};
    if (resident == nil || build.pipeline->states == nil ||
        std::any_of(terminals.begin(), terminals.end(),
                    [](const id<MTLBuffer> value) { return value == nil; })) {
      return rund::AccelCheck{false, "accel_metal_buffer_failed"};
    }
    [build.encoder setComputePipelineState:build.advance];
    [build.encoder setBuffer:terminals[0] offset:0u atIndex:0u];
    [build.encoder setBuffer:terminals[1] offset:0u atIndex:1u];
    [build.encoder setBuffer:terminals[2] offset:0u atIndex:2u];
    [build.encoder setBuffer:resident offset:0u atIndex:3u];
    [build.encoder setBuffer:build.pipeline->states offset:0u atIndex:4u];
    [build.encoder setBuffer:build.pipeline->control offset:0u atIndex:5u];
    MetalWindowParams params = route->params;
    [build.encoder setBytes:&params length:sizeof(params) atIndex:6u];
    [build.encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    const rund::AccelCheck capture = CheckMetalPipelineCapture(build.captured);
    if (!capture.ok) {
      return capture;
    }
    build.captured.commands.back().control = true;
    build.captured.commands.back().trace = false;
    if (build.advance_count == std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    ++build.advance_count;
    [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  if (encoded) {
    [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_program_internal
