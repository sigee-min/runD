#include "src/accel/metal/kernel/pipeline/icb.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstdint>
#include <cstdio>

namespace {

using namespace rund::node::accel::detail;

inline constexpr std::uint32_t BoundaryValue = 0x13579bdfu;
inline constexpr std::uint64_t BoundaryCommandCount =
    MetalPipelineIcbFullCommandCapacity + 1u;

struct NativePipelines final {
  id<MTLComputePipelineState> noop = nil;
  id<MTLComputePipelineState> write = nil;
  id<MTLComputePipelineState> read = nil;

  [[nodiscard]] bool valid() const noexcept {
    return noop != nil && write != nil && read != nil;
  }
};

[[nodiscard]] id<MTLComputePipelineState>
BuildPipeline(id<MTLDevice> const device, id<MTLFunction> const function,
              NSError **const failure) {
  if (device == nil || function == nil) {
    return nil;
  }
  MTLComputePipelineDescriptor *const descriptor =
      [[MTLComputePipelineDescriptor alloc] init];
  descriptor.computeFunction = function;
  descriptor.supportIndirectCommandBuffers = YES;
  return [device newComputePipelineStateWithDescriptor:descriptor
                                               options:MTLPipelineOptionNone
                                            reflection:nil
                                                 error:failure];
}

[[nodiscard]] NativePipelines BuildPipelines(id<MTLDevice> const device,
                                             NSError **const failure) {
  static constexpr const char Source[] = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void boundary_noop(uint lane [[thread_position_in_grid]]) {
  (void)lane;
}

kernel void boundary_write(device uint *state [[buffer(0)]],
                           uint lane [[thread_position_in_grid]]) {
  if (lane == 0u) {
    state[0] = 0x13579bdfu;
  }
}

kernel void boundary_read(device const uint *state [[buffer(0)]],
                          device uint *result [[buffer(1)]],
                          uint lane [[thread_position_in_grid]]) {
  if (lane == 0u) {
    result[0] = state[0];
  }
}
)metal";
  NSString *const source = [NSString stringWithUTF8String:Source];
  id<MTLLibrary> const library = [device newLibraryWithSource:source
                                                      options:nil
                                                        error:failure];
  if (library == nil) {
    return {};
  }
  id<MTLFunction> const noop = [library newFunctionWithName:@"boundary_noop"];
  id<MTLFunction> const write = [library newFunctionWithName:@"boundary_write"];
  id<MTLFunction> const read = [library newFunctionWithName:@"boundary_read"];
  if (noop == nil || write == nil || read == nil) {
    return {};
  }
  NativePipelines result{};
  result.noop = BuildPipeline(device, noop, failure);
  if (result.noop == nil) {
    return {};
  }
  result.write = BuildPipeline(device, write, failure);
  if (result.write == nil) {
    return {};
  }
  result.read = BuildPipeline(device, read, failure);
  return result;
}

inline void DispatchOne(id<MTLIndirectComputeCommand> const command,
                        id<MTLComputePipelineState> const pipeline) {
  [command setComputePipelineState:pipeline];
  [command concurrentDispatchThreads:MTLSizeMake(1u, 1u, 1u)
               threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
}

[[nodiscard]] const char *ErrorText(NSError *const error) noexcept {
  NSString *const text = error == nil ? nil : error.localizedDescription;
  const char *const utf8 = text == nil ? nullptr : text.UTF8String;
  return utf8 == nullptr ? "unknown" : utf8;
}

} // namespace

int main() {
  @autoreleasepool {
    const auto begin = std::chrono::steady_clock::now();
    id<MTLDevice> const device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> const queue =
        device == nil ? nil : [device newCommandQueue];
    NSError *failure = nil;
    const NativePipelines pipelines =
        device == nil ? NativePipelines{} : BuildPipelines(device, &failure);
    if (device == nil || queue == nil || !pipelines.valid()) {
      std::fprintf(stderr, "metal_icb_boundary,error,setup,%s\n",
                   ErrorText(failure));
      return 1;
    }

    id<MTLBuffer> const state =
        [device newBufferWithLength:sizeof(std::uint32_t)
                            options:MTLResourceStorageModeShared];
    id<MTLBuffer> const result =
        [device newBufferWithLength:sizeof(std::uint32_t)
                            options:MTLResourceStorageModeShared];
    id<MTLIndirectCommandBuffer> const full =
        AllocateMetalPipelineIcb(device, MetalPipelineIcbFullCommandCapacity);
    id<MTLIndirectCommandBuffer> const tail =
        AllocateMetalPipelineIcb(device, 1u);
    if (state == nil || result == nil || state.contents == nullptr ||
        result.contents == nullptr || full == nil || tail == nil ||
        full.size != MetalPipelineIcbFullCommandCapacity || tail.size != 1u) {
      std::fputs("metal_icb_boundary,error,allocation\n", stderr);
      return 2;
    }
    *static_cast<std::uint32_t *>(state.contents) = 0u;
    *static_cast<std::uint32_t *>(result.contents) = 0u;

    for (NSUInteger index = 0u;
         index + 1u < MetalPipelineIcbFullCommandCapacity; ++index) {
      id<MTLIndirectComputeCommand> const command =
          [full indirectComputeCommandAtIndex:index];
      if (command == nil) {
        std::fprintf(stderr, "metal_icb_boundary,error,command,%llu\n",
                     static_cast<unsigned long long>(index));
        return 3;
      }
      DispatchOne(command, pipelines.noop);
    }
    id<MTLIndirectComputeCommand> const writer = [full
        indirectComputeCommandAtIndex:MetalPipelineIcbFullCommandCapacity - 1u];
    id<MTLIndirectComputeCommand> const reader =
        [tail indirectComputeCommandAtIndex:0u];
    if (writer == nil || reader == nil) {
      std::fputs("metal_icb_boundary,error,boundary_command\n", stderr);
      return 4;
    }
    [writer setComputePipelineState:pipelines.write];
    [writer setKernelBuffer:state offset:0u atIndex:0u];
    [writer concurrentDispatchThreads:MTLSizeMake(1u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];
    [reader setComputePipelineState:pipelines.read];
    [reader setKernelBuffer:state offset:0u atIndex:0u];
    [reader setKernelBuffer:result offset:0u atIndex:1u];
    [reader concurrentDispatchThreads:MTLSizeMake(1u, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(1u, 1u, 1u)];

    const MetalIcbChunk chunks[]{
        MetalIcbChunk{
            .commands = full,
            .command_count =
                static_cast<std::uint32_t>(MetalPipelineIcbFullCommandCapacity),
        },
        MetalIcbChunk{
            .commands = tail,
            .command_count = 1u,
            .flags = MetalIcbChunkBarrierBefore,
        },
    };
    id<MTLCommandBuffer> const command =
        [queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> const encoder =
        command == nil ? nil : [command computeCommandEncoder];
    if (command == nil || encoder == nil) {
      std::fputs("metal_icb_boundary,error,submission\n", stderr);
      return 5;
    }
    const id<MTLResource> resources[]{state, result};
    [encoder useResources:resources
                    count:2u
                    usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    const rund::AccelCheck encoded =
        EncodeMetalPipelineIcbChunks(encoder, chunks, 2u);
    [encoder endEncoding];
    if (!encoded.ok) {
      std::fprintf(stderr, "metal_icb_boundary,error,encode,%s\n",
                   encoded.reason);
      return 6;
    }

    [command commit];
    [command waitUntilCompleted];
    const std::uint32_t observed =
        *static_cast<const std::uint32_t *>(result.contents);
    if (command.status != MTLCommandBufferStatusCompleted ||
        command.error != nil || observed != BoundaryValue) {
      std::fprintf(stderr, "metal_icb_boundary,error,execute,%s,observed,%u\n",
                   ErrorText(command.error), observed);
      return 7;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - begin);
    std::printf(
        "metal_icb_boundary,ok,commands,%llu,chunks,2,boundary,65536,"
        "boundary_barriers,1,command_submits,1,result,%u,full_bytes,%llu,"
        "tail_bytes,%llu,elapsed_us,%lld\n",
        static_cast<unsigned long long>(BoundaryCommandCount), observed,
        static_cast<unsigned long long>(full.allocatedSize),
        static_cast<unsigned long long>(tail.allocatedSize),
        static_cast<long long>(elapsed.count()));
    return 0;
  }
}
