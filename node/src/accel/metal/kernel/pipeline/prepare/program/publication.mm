#include "internal.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::node::accel::detail::metal_pipeline_program_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck EncodePublications(MetalPipelineBuild &build,
                                    const ProgramEntry &entry) {
  if (entry.resident_window != nullptr &&
      entry.resident_window->phase == BackendWindowPhase::NestedFold) {
    for (const MetalPublish &publication : build.native_publication_rows()) {
      if (publication.params.kind !=
              static_cast<std::uint32_t>(
                  PreparedKernelPublicationKind::Window) ||
          publication.params.state != entry.resident_window->state ||
          publication.params.tile == 0u) {
        continue;
      }
      id<MTLBuffer> const source =
          (__bridge id<MTLBuffer>)publication.sources[0].get();
      id<MTLBuffer> const target =
          (__bridge id<MTLBuffer>)publication.target.get();
      id<MTLBuffer> const count =
          (__bridge id<MTLBuffer>)publication.count.get();
      if (source == nil || target == nil || count == nil ||
          build.pipeline->states == nil) {
        return rund::AccelCheck{false, "accel_metal_buffer_failed"};
      }
      MetalPublishParams params = publication.params;
      params.outer = entry.resident_window->outer_iteration;
      [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [build.encoder setComputePipelineState:build.publish];
      [build.encoder setBuffer:source offset:0u atIndex:0u];
      [build.encoder setBuffer:source offset:0u atIndex:1u];
      [build.encoder setBuffer:source offset:0u atIndex:2u];
      [build.encoder setBuffer:target offset:0u atIndex:3u];
      [build.encoder setBuffer:build.pipeline->control offset:0u atIndex:4u];
      [build.encoder setBuffer:build.pipeline->states offset:0u atIndex:5u];
      [build.encoder setBytes:&params length:sizeof(params) atIndex:6u];
      [build.encoder setBuffer:count offset:0u atIndex:7u];
      const NSUInteger count_threads =
          static_cast<NSUInteger>(publication.params.tile);
      const NSUInteger width = std::min(
          count_threads, [build.publish maxTotalThreadsPerThreadgroup]);
      build.captured.owner = std::numeric_limits<std::uint32_t>::max();
      [build.encoder dispatchThreads:MTLSizeMake(count_threads, 1u, 1u)
               threadsPerThreadgroup:MTLSizeMake(width, 1u, 1u)];
      const rund::AccelCheck capture =
          CheckMetalPipelineCapture(build.captured);
      if (!capture.ok) {
        return capture;
      }
      build.captured.commands.back().control = true;
      build.captured.commands.back().trace = true;
      ++build.pipeline->dispatch_count;
      ++build.window_publish_count;
      [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  }
  if (entry.resident_window != nullptr &&
      entry.resident_window->advances_outer_state() &&
      entry.resident_window->outer_iteration + 1u ==
          entry.resident_window->outer_bound) {
    for (const MetalPublish &publication : build.native_publication_rows()) {
      if (publication.params.kind !=
              static_cast<std::uint32_t>(
                  PreparedKernelPublicationKind::Terminal) ||
          publication.params.state != entry.resident_window->state ||
          publication.params.final > 2u || publication.params.count == 0u) {
        continue;
      }
      const std::uint32_t final = publication.params.final;
      id<MTLBuffer> const target =
          (__bridge id<MTLBuffer>)publication.sources[final].get();
      std::array<id<MTLBuffer>, 3u> sources{
          (__bridge id<MTLBuffer>)publication.sources[0].get(),
          (__bridge id<MTLBuffer>)publication.sources[1].get(),
          (__bridge id<MTLBuffer>)publication.sources[2].get()};
      if (target == nil || build.pipeline->states == nil ||
          std::any_of(sources.begin(), sources.end(),
                      [](const id<MTLBuffer> value) { return value == nil; })) {
        return rund::AccelCheck{false, "accel_metal_buffer_failed"};
      }
      MetalPublishParams params = publication.params;
      params.target_offset_words = params.source_offset_words[final];
      params.target_stride_words = params.source_stride_words[final];
      params.stop = std::numeric_limits<std::uint32_t>::max();
      [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      [build.encoder setComputePipelineState:build.publish];
      [build.encoder setBuffer:sources[0] offset:0u atIndex:0u];
      [build.encoder setBuffer:sources[1] offset:0u atIndex:1u];
      [build.encoder setBuffer:sources[2] offset:0u atIndex:2u];
      [build.encoder setBuffer:target offset:0u atIndex:3u];
      [build.encoder setBuffer:build.pipeline->control offset:0u atIndex:4u];
      [build.encoder setBuffer:build.pipeline->states offset:0u atIndex:5u];
      [build.encoder setBytes:&params length:sizeof(params) atIndex:6u];
      [build.encoder setBuffer:build.pipeline->control offset:0u atIndex:7u];
      const NSUInteger count =
          static_cast<NSUInteger>(publication.params.count);
      const NSUInteger width =
          std::min(count, [build.publish maxTotalThreadsPerThreadgroup]);
      build.captured.owner = std::numeric_limits<std::uint32_t>::max();
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
      ++build.canonicalize_count;
      [build.encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
  }
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::metal_pipeline_program_internal
