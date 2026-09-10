#include "compile.hpp"
#include "../../object.hpp"
#include "../../state.hpp"
#include "../named.hpp"
#include <memory>
#include <string>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#endif

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

namespace {
void AppendMetalArtifactHex64(std::string &out, const rund::kernel::u64 value) {
  constexpr char kHex[] = "0123456789abcdef";
  for (int shift = 60; shift >= 0; shift -= 4) {
    out.push_back(kHex[(value >> static_cast<unsigned>(shift)) & 0xfu]);
  }
}

[[nodiscard]] NSString *
MetalMapArtifactFunctionName(const rund::kernel::ArtifactKey &key) {
  std::string name = "rund_compute_map_";
  AppendMetalArtifactHex64(name, key.op_hash_hi);
  name.push_back('_');
  AppendMetalArtifactHex64(name, key.op_hash_lo);
  if (key.variant == rund::kernel::LoweringArtifactVariant::Controlled) {
    name += "_controlled";
  } else if (key.variant == rund::kernel::LoweringArtifactVariant::Recurrence) {
    name += "_recurrence";
  } else if (key.variant ==
             rund::kernel::LoweringArtifactVariant::HistoryRecurrence) {
    name += "_history_recurrence";
  } else if (key.variant == rund::kernel::LoweringArtifactVariant::DeviceVsm) {
    name += "_device_vsm";
  }
  return [NSString stringWithUTF8String:name.c_str()];
}
} // namespace

[[nodiscard]] std::shared_ptr<void> CompileMetalMapArtifactPipeline(
    MetalAdapter &adapter, const rund::kernel::LoweringArtifact &artifact) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    SetMetalLastError(adapter, "accel_metal_unavailable");
    return {};
  }
  NSString *const source =
      [[NSString alloc] initWithBytes:artifact.source_text.data()
                               length:artifact.source_text.size()
                             encoding:NSUTF8StringEncoding];
  if (source == nil) {
    SetMetalLastError(adapter, "compute_artifact_source_invalid");
    return {};
  }
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source
                                                options:nil
                                                  error:&error];
  if (library == nil) {
    SetMetalLastErrorDetail(
        adapter, error == nil ? "accel_metal_source_compile_failed"
                              : [[error localizedDescription] UTF8String]);
    return {};
  }
  id<MTLFunction> function =
      [library newFunctionWithName:MetalMapArtifactFunctionName(artifact.key)];
  if (function == nil) {
    SetMetalLastError(adapter, "compute_artifact_function_unavailable");
    return {};
  }
  id<MTLComputePipelineState> pipeline =
      NewMetalPipeline(device, function, &error);
  if (pipeline == nil) {
    SetMetalLastErrorDetail(
        adapter, error == nil ? "accel_metal_pipeline_unavailable"
                              : [[error localizedDescription] UTF8String]);
    return {};
  }
  SetMetalLastError(adapter, "ok");
  return RetainMetalObject((__bridge void *)pipeline);
}
#endif

} // namespace rund::node::accel::detail
