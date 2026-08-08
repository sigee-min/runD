#pragma once

#include <accel/check.hpp>

#include "lookup.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
PrepareMetalStencilPipeline(MetalAdapter &adapter,
                            const rund::kernel::StencilPlan &plan,
                            const rund::kernel::ComputeDomain domain,
                            const StencilGpuShapeCandidates &candidates,
                            MetalStencilEncodeResources &resources) {
  resources.pipeline.reset();
  MetalStencilPipelineAttempt failed{};
  const StencilGpuShapeSelection selection = SelectStencilGpuShapeCandidate(
      candidates, [&](const StencilGpuShape shape) {
        std::shared_ptr<void> pipeline;
        const MetalStencilPipelineAttempt attempt = CompileMetalStencilPipeline(
            adapter, plan.op, plan.element, domain, shape, pipeline);
        if (attempt.status == MetalStencilPipelineAttemptStatus::Unsupported) {
          return StencilGpuCandidateDecision::Skip;
        }
        if (attempt.status == MetalStencilPipelineAttemptStatus::Failed) {
          failed = attempt;
          return StencilGpuCandidateDecision::Abort;
        }
        resources.pipeline = std::move(pipeline);
        return StencilGpuCandidateDecision::Select;
      });
  if (selection.aborted) {
    resources.pipeline.reset();
    return rund::AccelCheck{false, failed.reason};
  }
  resources.shape = selection.shape;
  if (!resources.shape.valid()) {
    resources.pipeline.reset();
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
