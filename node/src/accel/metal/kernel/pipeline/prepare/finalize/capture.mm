#include "internal.hpp"

#include <algorithm>
#include <array>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck FinalizeMetalCapture(MetalPipelineBuild &build) {
  build.captured.replacements = {};
  build.captured.replacement_target = nil;
  if (build.captured.commands.size() == build.reset_command_count) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  if (build.aggregate_selected) {
    if (build.captured.commands.size() != 2u) {
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    build.pipeline->control_command_count = 1u;
  } else {
    // The final captured producer also needs an ICB barrier: command-buffer
    // completion alone does not publish a concurrent indirect dispatch's
    // writes to later readback command buffers.
    [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    build.pipeline->control_command_count =
        2u + build.import_count + static_cast<std::uint32_t>(build.needs_reset) +
        static_cast<std::uint32_t>(build.pipeline->telemetry.size()) +
        build.fold_count + build.advance_count + build.canonicalize_count +
        build.window_publish_count;
    [build.encoder setComputePipelineState:build.complete];
    [build.encoder setBuffer:build.pipeline->control offset:0u atIndex:0u];
    [build.encoder setBytes:&build.status_params
                      length:sizeof(build.status_params)
                     atIndex:1u];
    if (build.profile_steps) {
      [build.encoder setBuffer:build.pipeline->step_control offset:0u atIndex:2u];
    }
    id<MTLBuffer> const states =
        build.pipeline->states == nil ? build.pipeline->control
                                      : build.pipeline->states;
    [build.encoder setBuffer:states offset:0u atIndex:3u];
    [build.encoder dispatchThreads:MTLSizeMake(1u, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    const rund::AccelCheck completed =
        CheckMetalPipelineCapture(build.captured);
    if (!completed.ok) {
      return completed;
    }
    build.captured.commands.back().control = true;
    build.captured.commands.back().trace = false;
    if (build.native_publication_count != 0u) {
      [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      for (const MetalPublish &publication : build.native_publication_rows()) {
        if (publication.params.kind !=
                static_cast<std::uint32_t>(
                    PreparedKernelPublicationKind::Terminal) ||
            publication.params.count == 0u) {
          continue;
        }
        id<MTLBuffer> const target =
            (__bridge id<MTLBuffer>)publication.target.get();
        std::array<id<MTLBuffer>, 3u> sources{
            (__bridge id<MTLBuffer>)publication.sources[0].get(),
            (__bridge id<MTLBuffer>)publication.sources[1].get(),
            (__bridge id<MTLBuffer>)publication.sources[2].get()};
        if (target == nil || build.pipeline->states == nil ||
            std::any_of(
                sources.begin(), sources.end(),
                [](const id<MTLBuffer> value) { return value == nil; })) {
          return rund::AccelCheck{false, "accel_metal_buffer_failed"};
        }
        [build.encoder setComputePipelineState:build.publish];
        [build.encoder setBuffer:sources[0] offset:0u atIndex:0u];
        [build.encoder setBuffer:sources[1] offset:0u atIndex:1u];
        [build.encoder setBuffer:sources[2] offset:0u atIndex:2u];
        [build.encoder setBuffer:target offset:0u atIndex:3u];
        [build.encoder setBuffer:build.pipeline->control offset:0u atIndex:4u];
        [build.encoder setBuffer:build.pipeline->states offset:0u atIndex:5u];
        [build.encoder setBytes:&publication.params
                          length:sizeof(publication.params)
                         atIndex:6u];
        [build.encoder setBuffer:build.pipeline->control offset:0u atIndex:7u];
        const NSUInteger count =
            static_cast<NSUInteger>(publication.params.count);
        const NSUInteger width =
            std::min(count, [build.publish maxTotalThreadsPerThreadgroup]);
        [build.encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
            threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
        const rund::AccelCheck capture =
            CheckMetalPipelineCapture(build.captured);
        if (!capture.ok) {
          return capture;
        }
        build.captured.commands.back().control = true;
        build.captured.commands.back().trace = true;
        ++build.pipeline->dispatch_count;
        ++build.pipeline->control_command_count;
      }
      [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  }
  const rund::AccelCheck capture = CheckMetalPipelineCapture(build.captured);
  if (!capture.ok) {
    return capture;
  }
  if (build.captured.commands.empty()) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  build.pipeline->dispatch_count = static_cast<std::uint64_t>(
      std::count_if(build.captured.commands.begin(),
                    build.captured.commands.end(),
                    [](const MetalCommand &command) { return command.trace; }));
  if (build.pipeline->dispatch_count == 0u) {
    return rund::AccelCheck{false, "accel_kernel_primitive_unsupported"};
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
