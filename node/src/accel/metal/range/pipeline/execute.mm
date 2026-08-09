#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../../kernel/backend/run.hpp"
#include "../../../kernel/preparation.hpp"
#include "../../../kernel/scratch.hpp"
#include "../../../range_aggregate/plan.hpp"
#include "../../pipeline/template.hpp"
#include "../../scratch.hpp"
#include "../encode/dispatch.hpp"
#include "../local.hpp"
#include "../resources/prepare.hpp"
#include "store.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace rund::node::accel::detail {

MetalRangeAttempt CompileMetalRange(MetalAdapter &adapter,
                                    const RangeExec &execution,
                                    std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.reset();
  const std::string key = RangePipelineKey(execution);
  if (LookupMetalRangePipeline(adapter, key, out)) {
    const MetalRangeAssessment assessment =
        AssessMetalRange(adapter, execution, out);
    if (assessment == MetalRangeAssessment::Ready) {
      return {MetalRangeAttemptStatus::Ready, "ok"};
    }
    out.reset();
    return {assessment == MetalRangeAssessment::Unsupported
                ? MetalRangeAttemptStatus::Unsupported
                : MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return {MetalRangeAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  const MetalRangeAttempt compiled =
      CompileMetalRangeLibrary(adapter, execution, out);
  if (compiled.status != MetalRangeAttemptStatus::Ready) {
    return compiled;
  }
  const MetalNamedPipelinePublishResult published =
      StoreMetalRangePipeline(adapter, key, out, compiled.create_ns);
  if (published.status != MetalNamedPipelinePublishStatus::Inserted) {
    // This invocation constructed a PSO, so account it even when a concurrent
    // owner won publication or the cache could not publish it. Inserted PSOs
    // are accounted by PublishMetalNamedPipeline itself.
    RecordMetalUncachedPipelineCompile(adapter, compiled.create_ns);
  }
  if (published.status == MetalNamedPipelinePublishStatus::Failed) {
    out.reset();
    return {MetalRangeAttemptStatus::Failed, "compute_pipeline_capacity"};
  }
  out = published.pipeline;
  const MetalRangeAssessment assessment =
      AssessMetalRange(adapter, execution, out);
  if (assessment == MetalRangeAssessment::Ready) {
    return compiled;
  }
  out.reset();
  return {assessment == MetalRangeAssessment::Unsupported
              ? MetalRangeAttemptStatus::Unsupported
              : MetalRangeAttemptStatus::Failed,
          "accel_metal_pipeline_unavailable"};
#else
  (void)adapter;
  (void)execution;
  (void)out;
  return {MetalRangeAttemptStatus::Failed, "accel_metal_pipeline_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
