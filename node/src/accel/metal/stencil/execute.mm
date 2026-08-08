#include <accel/check.hpp>
#include <accel/device.hpp>

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#import <Metal/Metal.h>
#endif

#include "../../stencil/shape.hpp"
#include "../command/run.hpp"
#include "../pipeline/template.hpp"
#include "encode/dispatch.hpp"
#include "local.hpp"
#include "pipeline/store.hpp"
#include "resources/pipeline.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] StencilGpuCapabilities
MetalStencilCapabilities(id<MTLDevice> device) noexcept {
  if (device == nil) {
    return {};
  }
  const std::uint64_t maximum_width = device.maxThreadsPerThreadgroup.width;
  return StencilGpuCapabilities{
      .maximum_workgroup_width =
          static_cast<rund::kernel::u32>(std::min<std::uint64_t>(
              maximum_width, std::numeric_limits<rund::kernel::u32>::max())),
      .shared_memory_occupancy_budget = kStencilSharedMemoryOccupancyBudget,
      .shared_memory_limit = device.maxThreadgroupMemoryLength,
      .maximum_group_count = std::numeric_limits<std::uint32_t>::max(),
  };
}

[[nodiscard]] bool MetalStencilDispatchCanFitAnyWidth(
    const rund::kernel::StencilPlan &plan,
    const StencilGpuCapabilities capabilities) noexcept {
  StencilGpuShape widest{};
  for (const rund::kernel::u32 width : kStencilPhysicalGroupWidths) {
    if (width <= capabilities.maximum_workgroup_width) {
      widest = StencilGpuShape::direct(width);
    }
  }
  return widest.valid() &&
         StencilPhysicalGroupsFit(plan.element_count,
                                  capabilities.maximum_group_count, widest);
}

[[nodiscard]] bool MetalStencilHasSupportedWidth(
    const StencilGpuCapabilities capabilities) noexcept {
  return capabilities.maximum_workgroup_width >=
         kStencilPhysicalGroupWidths.front();
}

} // namespace
#endif

void DestroyMetalStencilEncodeResources(void *const raw) {
  auto *const resources = static_cast<MetalStencilEncodeResources *>(raw);
  delete resources;
}

MetalStencilPipelineAttempt CompileMetalStencilPipeline(
    MetalAdapter &adapter, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element,
    const rund::kernel::ComputeDomain domain, const StencilGpuShape shape,
    std::shared_ptr<void> &out) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  out.reset();
  const std::string key = StencilPipelineKey(op, element, domain, shape);
  if (LookupMetalStencilPipeline(adapter, key, out)) {
    const MetalStencilPipelineAssessment assessment =
        AssessMetalStencilPipeline(adapter, op, element, domain, shape, out);
    if (assessment == MetalStencilPipelineAssessment::Ready) {
      return {MetalStencilPipelineAttemptStatus::Ready, "ok"};
    }
    out.reset();
    return {assessment == MetalStencilPipelineAssessment::Unsupported
                ? MetalStencilPipelineAttemptStatus::Unsupported
                : MetalStencilPipelineAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter.device.get();
  if (device == nil) {
    return {MetalStencilPipelineAttemptStatus::Failed,
            "accel_metal_pipeline_unavailable"};
  }
  const MetalStencilPipelineAttempt compiled =
      CompileMetalStencilPipelineLibrary(adapter, op, element, domain, shape,
                                         out);
  if (compiled.status != MetalStencilPipelineAttemptStatus::Ready) {
    return compiled;
  }
  const MetalNamedPipelinePublishResult published =
      StoreMetalStencilPipeline(adapter, key, out, compiled.create_ns);
  if (published.status != MetalNamedPipelinePublishStatus::Inserted) {
    // This invocation constructed a PSO, so account it even when a concurrent
    // owner won publication or the cache could not publish it. Inserted PSOs
    // are accounted by PublishMetalNamedPipeline itself.
    RecordMetalUncachedPipelineCompile(adapter, compiled.create_ns);
  }
  if (published.status == MetalNamedPipelinePublishStatus::Failed) {
    out.reset();
    return {MetalStencilPipelineAttemptStatus::Failed,
            "compute_pipeline_capacity"};
  }
  out = published.pipeline;
  const MetalStencilPipelineAssessment assessment =
      AssessMetalStencilPipeline(adapter, op, element, domain, shape, out);
  if (assessment == MetalStencilPipelineAssessment::Ready) {
    return compiled;
  }
  out.reset();
  return {assessment == MetalStencilPipelineAssessment::Unsupported
              ? MetalStencilPipelineAttemptStatus::Unsupported
              : MetalStencilPipelineAttemptStatus::Failed,
          "accel_metal_pipeline_unavailable"};
#else
  (void)adapter;
  (void)op;
  (void)element;
  (void)domain;
  (void)shape;
  (void)out;
  return {MetalStencilPipelineAttemptStatus::Failed,
          "accel_metal_pipeline_unavailable"};
#endif
}

rund::AccelCheck PrepareMetalStencil(
    const rund::AccelDevice &pick, const rund::kernel::StencilDesc &desc,
    const rund::kernel::StencilPlan &plan,
    const rund::kernel::ComputeDomain domain, const StencilBinds &bindings,
    std::shared_ptr<void> &resources,
    const MetalKernelImmutablePipelines *const pipelines) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  resources.reset();
  if (!MetalPickOwnsAdapter(pick)) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  SetMetalLastError(*adapter, "ok");
  if (!StencilShapeOk(desc, plan, bindings)) {
    SetMetalLastError(*adapter, "compute_stencil_invalid");
    return rund::AccelCheck{false, "compute_stencil_invalid"};
  }
  id<MTLDevice> device = (__bridge id<MTLDevice>)adapter->device.get();
  const StencilGpuCapabilities capabilities = MetalStencilCapabilities(device);
  const StencilGpuShapeCandidates candidates =
      RankStencilGpuShapes(plan.element_count, plan.radius,
                           StencilElementBytes(plan.element), capabilities);
  if (candidates.empty()) {
    const char *const reason =
        MetalStencilHasSupportedWidth(capabilities) &&
                !MetalStencilDispatchCanFitAnyWidth(plan, capabilities)
            ? "compute_dispatch_overflow"
            : "accel_metal_pipeline_unavailable";
    SetMetalLastError(*adapter, reason);
    return rund::AccelCheck{false, reason};
  }

  auto *const raw = new MetalStencilEncodeResources{};
  std::shared_ptr<void> owned{raw, DestroyMetalStencilEncodeResources};
  raw->adapter = adapter;
  raw->plan = plan;
  rund::AccelCheck check =
      LookupMetalStencilResidentBuffers(pick, bindings, *raw);
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
  }
  if (pipelines != nullptr && pipelines->ready(1u)) {
    raw->pipeline = pipelines->stages[0u];
    const StencilGpuShapeSelection selection = SelectStencilGpuShapeCandidate(
        candidates, [&](const StencilGpuShape shape) {
          const MetalStencilPipelineAssessment assessment =
              AssessMetalStencilPipeline(*adapter, plan.op, plan.element,
                                         domain, shape, raw->pipeline);
          if (assessment == MetalStencilPipelineAssessment::DifferentShape) {
            return StencilGpuCandidateDecision::Skip;
          }
          return assessment == MetalStencilPipelineAssessment::Ready
                     ? StencilGpuCandidateDecision::Select
                     : StencilGpuCandidateDecision::Abort;
        });
    raw->shape = selection.shape;
    if (selection.aborted || !raw->shape.valid()) {
      check = {false, "accel_metal_pipeline_unavailable"};
    }
  } else if (pipelines != nullptr) {
    check = {false, "accel_metal_pipeline_unavailable"};
  } else {
    check =
        PrepareMetalStencilPipeline(*adapter, plan, domain, candidates, *raw);
  }
  if (!check.ok) {
    SetMetalLastError(*adapter, check.reason);
    return check;
  }
  resources = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  (void)resources;
  (void)pipelines;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck EncodeMetalStencil(MetalAdapter &adapter,
                                    const std::shared_ptr<void> &resources,
                                    void *command_encoder) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalStencilCommandState state{};
  const rund::AccelCheck prepared = PrepareMetalStencilCommandState(
      adapter, resources, command_encoder, state);
  if (!prepared.ok) {
    return prepared;
  }
  const StencilParams params{state.stencil->plan.element_count,
                             state.stencil->plan.radius};
  EncodeMetalStencilDispatch(state, params);
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_encoder;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

rund::AccelCheck ExecuteMetalStencil(const rund::AccelDevice &pick,
                                     const rund::kernel::StencilDesc &desc,
                                     const rund::kernel::StencilPlan &plan,
                                     const rund::kernel::ComputeDomain domain,
                                     const StencilBinds &bindings) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  auto *const adapter = static_cast<MetalAdapter *>(pick.backend.context);
  if (!MetalPickOwnsAdapter(pick) || adapter == nullptr ||
      adapter->queue == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  std::shared_ptr<void> resources{};
  const rund::AccelCheck prepare =
      PrepareMetalStencil(pick, desc, plan, domain, bindings, resources);
  if (!prepare.ok) {
    return prepare;
  }

  CommandRun command{};
  const rund::AccelCheck open = OpenCommand(*adapter, command);
  if (!open.ok) {
    return open;
  }
  const rund::AccelCheck encode =
      EncodeMetalStencil(*adapter, resources, (__bridge void *)command.encoder);
  const rund::AccelCheck submit = FinishCommand(*adapter, command, encode);
  if (!submit.ok) {
    return submit;
  }
  return FinishMetalStencil(*adapter, resources);
#else
  (void)pick;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)bindings;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
