#pragma once

#include "name.hpp"
#include "../named.hpp"
#include "../../object.hpp"
#include "../../state.hpp"
#include <memory>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

[[nodiscard]] std::shared_ptr<void> CompileMetalMapArtifactPipeline(
    MetalAdapter& adapter,
    const rund::kernel::LoweringArtifact& artifact) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return {};
  }
  NSString* const source =
      [[NSString alloc] initWithBytes:artifact.source_text.data()
                               length:artifact.source_text.size()
                             encoding:NSUTF8StringEncoding];
  if (source == nil) {
    return {};
  }
  NSError* error = nil;
  id<MTLLibrary> library =
      [device newLibraryWithSource:source options:nil error:&error];
  if (library == nil) {
    return {};
  }
  id<MTLFunction> function =
      [library newFunctionWithName:MetalMapArtifactFunctionName(artifact.key)];
  if (function == nil) {
    return {};
  }
  id<MTLComputePipelineState> pipeline =
      NewMetalPipeline(device, function, &error);
  return RetainMetalObject((__bridge void*)pipeline);
}

#endif

}  // namespace rund::node::accel::detail
