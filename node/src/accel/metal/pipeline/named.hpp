#pragma once

#include "../../clock.hpp"
#include "../object.hpp"
#include "cache.hpp"
#include "guard.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline id<MTLComputePipelineState>
NewMetalPipeline(id<MTLDevice> device, id<MTLFunction> function,
                 NSError **error) {
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
                                                 error:error];
}

[[nodiscard]] inline id<MTLLibrary>
NewMetalLibrary(id<MTLDevice> device, const std::string &source_text) {
  if (device == nil) {
    return nil;
  }
  NSString *const source =
      [[NSString alloc] initWithBytes:source_text.data()
                               length:source_text.size()
                             encoding:NSUTF8StringEncoding];
  if (source == nil) {
    return nil;
  }
  NSError *error = nil;
  id<MTLLibrary> library =
      [device newLibraryWithSource:source options:nil error:&error];
  return library;
}

[[nodiscard]] inline std::shared_ptr<void>
AcquireMetalLibrary(MetalAdapter &adapter, std::string source_text) {
  source_text = PipelinePrivateMetalSource(std::move(source_text));
  if (source_text.empty()) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return {};
  }
  std::shared_ptr<void> cached = LookupMetalSourceLibrary(adapter, source_text);
  if (cached != nullptr) {
    return cached;
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  const std::uint64_t begin = MonotonicNanoseconds();
  id<MTLLibrary> library = NewMetalLibrary(device, source_text);
  std::shared_ptr<void> owner = RetainMetalObject((__bridge void *)library);
  return PublishMetalSourceLibrary(adapter, std::move(source_text),
                                   std::move(owner),
                                   MonotonicNanoseconds() - begin);
}

[[nodiscard]] inline bool
MakeNamedMetalPipeline(id<MTLDevice> device, id<MTLLibrary> library,
                       NSString *const function_name,
                       const std::span<const std::uint32_t> constants,
                       std::shared_ptr<void> &out) {
  if (device == nil || library == nil || function_name == nil) {
    return false;
  }
  MTLFunctionConstantValues *const values =
      [[MTLFunctionConstantValues alloc] init];
  for (std::size_t index = 0u; index < constants.size(); ++index) {
    const std::uint32_t value = constants[index];
    [values setConstantValue:&value type:MTLDataTypeUInt atIndex:index];
  }
  NSError *function_error = nil;
  id<MTLFunction> function = [library newFunctionWithName:function_name
                                           constantValues:values
                                                    error:&function_error];
  if (function == nil) {
    return false;
  }
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline =
      NewMetalPipeline(device, function, &error);
  out = RetainMetalObject((__bridge void *)pipeline);
  return out != nullptr;
}

[[nodiscard]] inline bool MakeNamedMetalPipeline(id<MTLDevice> device,
                                                 id<MTLLibrary> library,
                                                 NSString *const function_name,
                                                 std::shared_ptr<void> &out) {
  return MakeNamedMetalPipeline(device, library, function_name, {}, out);
}

[[nodiscard]] inline bool
MakeNamedMetalPipeline(id<MTLDevice> device, id<MTLLibrary> library,
                       const char *const function_name,
                       std::shared_ptr<void> &out) {
  if (function_name == nullptr) {
    return false;
  }
  return MakeNamedMetalPipeline(
      device, library, [NSString stringWithUTF8String:function_name], out);
}

[[nodiscard]] inline bool
MakeNamedMetalPipeline(id<MTLDevice> device, id<MTLLibrary> library,
                       const char *const function_name,
                       const std::span<const std::uint32_t> constants,
                       std::shared_ptr<void> &out) {
  if (function_name == nullptr) {
    return false;
  }
  return MakeNamedMetalPipeline(device, library,
                                [NSString stringWithUTF8String:function_name],
                                constants, out);
}
#endif

} // namespace rund::node::accel::detail
