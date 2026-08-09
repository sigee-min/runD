#pragma once

#include "../../pipeline/named.hpp"
#include "cache.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
enum class MetalRangeAssessment : std::uint8_t {
  Ready,
  DifferentShape,
  Unsupported,
  Invalid,
};

[[nodiscard]] inline std::string MetalRangeLabel(const RangeExec &execution) {
  return MetalPipelineCacheKey(RangePipelineKey(execution));
}

[[nodiscard]] inline MetalRangeAssessment
AssessMetalRange(MetalAdapter &adapter, const RangeExec &execution,
                 const std::shared_ptr<void> &owner) {
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  id<MTLComputePipelineState> pipeline =
      (__bridge id<MTLComputePipelineState>)owner.get();
  if (device == nil || pipeline == nil || pipeline.device != device) {
    return MetalRangeAssessment::Invalid;
  }

  const std::string identity = MetalRangeLabel(execution);
  NSString *const label = [[NSString alloc] initWithBytes:identity.data()
                                                   length:identity.size()
                                                 encoding:NSUTF8StringEncoding];
  if (label == nil || pipeline.label == nil) {
    return MetalRangeAssessment::Invalid;
  }
  if (![pipeline.label isEqualToString:label]) {
    return MetalRangeAssessment::DifferentShape;
  }

  const std::uint64_t maximum_width =
      std::min<std::uint64_t>(device.maxThreadsPerThreadgroup.width,
                              pipeline.maxTotalThreadsPerThreadgroup);
  const MetalRangeSupport support = MetalRangeSupports(
      execution,
      MetalRangeLimits{
          .maximum_workgroup_width =
              static_cast<rund::kernel::u32>(std::min<std::uint64_t>(
                  maximum_width,
                  std::numeric_limits<rund::kernel::u32>::max())),
          .static_shared_bytes = pipeline.staticThreadgroupMemoryLength,
          .shared_memory_limit = device.maxThreadgroupMemoryLength,
      });
  if (support == MetalRangeSupport::Supported) {
    return MetalRangeAssessment::Ready;
  }
  return support == MetalRangeSupport::Unsupported
             ? MetalRangeAssessment::Unsupported
             : MetalRangeAssessment::Invalid;
}

[[nodiscard]] inline MetalRangeAttempt
CompileMetalRangeLibrary(MetalAdapter &adapter, const RangeExec &execution,
                         std::shared_ptr<void> &out) {
  out.reset();
  std::string source = PipelinePrivateMetalSource(MetalRangeSource(execution));
  if (source.empty()) {
    return {MetalRangeAttemptStatus::Failed, "compute_pipeline_capacity"};
  }
  std::shared_ptr<void> library_owner =
      LookupMetalSourceLibrary(adapter, source);
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return {MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  bool publish_library = false;
  std::uint64_t library_compile_ns = 0u;
  if (library_owner == nullptr) {
    const std::uint64_t begin = MonotonicNanoseconds();
    id<MTLLibrary> candidate = NewMetalLibrary(device, source);
    library_compile_ns = MonotonicNanoseconds() - begin;
    library_owner = RetainMetalObject((__bridge void *)candidate);
    publish_library = true;
  }
  const auto record_unpublished_library = [&]() noexcept {
    if (publish_library) {
      RecordMetalUncachedLibraryCompile(adapter, library_compile_ns);
      publish_library = false;
    }
  };
  id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_owner.get();
  const std::string function_name = RangeFunctionName(execution);
  const std::string identity = MetalRangeLabel(execution);
  NSString *const function =
      [[NSString alloc] initWithBytes:function_name.data()
                               length:function_name.size()
                             encoding:NSUTF8StringEncoding];
  NSString *const label = [[NSString alloc] initWithBytes:identity.data()
                                                   length:identity.size()
                                                 encoding:NSUTF8StringEncoding];
  if (library == nil || function == nil || label == nil) {
    record_unpublished_library();
    return {MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }

  id<MTLFunction> native_function = [library newFunctionWithName:function];
  if (native_function == nil) {
    record_unpublished_library();
    return {MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  MTLComputePipelineDescriptor *const descriptor =
      [[MTLComputePipelineDescriptor alloc] init];
  descriptor.label = label;
  descriptor.computeFunction = native_function;
  descriptor.supportIndirectCommandBuffers = YES;
  NSError *error = nil;
  const std::uint64_t pipeline_begin = MonotonicNanoseconds();
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithDescriptor:descriptor
                                            options:MTLPipelineOptionNone
                                         reflection:nil
                                              error:&error];
  const std::uint64_t pipeline_create_ns =
      MonotonicNanoseconds() - pipeline_begin;
  (void)error;
  out = RetainMetalObject((__bridge void *)pipeline);
  if (out == nullptr) {
    RecordMetalUncachedPipelineCompile(adapter, pipeline_create_ns);
    record_unpublished_library();
    return {MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  const MetalRangeAssessment assessment =
      AssessMetalRange(adapter, execution, out);
  if (assessment != MetalRangeAssessment::Ready) {
    RecordMetalUncachedPipelineCompile(adapter, pipeline_create_ns);
    record_unpublished_library();
    out.reset();
    return {assessment == MetalRangeAssessment::Unsupported
                ? MetalRangeAttemptStatus::Unsupported
                : MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  if (publish_library) {
    const MetalSourceLibraryPublishResult published = PublishMetalSourceLibrary(
        adapter, std::move(source), library_owner, library_compile_ns);
    // Publication accounts this constructed library for every disposition.
    // Do not let the earlier unpublished-library fallback count it again.
    publish_library = false;
    if (published.status == MetalSourceLibraryPublishStatus::Failed) {
      RecordMetalUncachedPipelineCompile(adapter, pipeline_create_ns);
      out.reset();
      return {MetalRangeAttemptStatus::Failed, "compute_pipeline_capacity"};
    }
    library_owner = published.library;
  }
  return {MetalRangeAttemptStatus::Ready, "ok", pipeline_create_ns};
}
#endif

} // namespace rund::node::accel::detail
