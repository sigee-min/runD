#include "internal.hpp"

#include "../../buffer/owner.hpp"
#include "../../state.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

rund::AccelCheck PrepareMetalViewLowering(
    const rund::AccelDevice &pick, const BoundStep &source,
    const KernelPreparationMode mode, const KernelViewLayout *const views,
    const RunBinds *const view_binds, std::shared_ptr<MetalViewLowering> &out) {
  out.reset();
  if (!MetalViewRequiresLowering(source)) {
    return rund::AccelCheck{true, "ok"};
  }

  std::shared_ptr<MetalViewLowering> candidate;
  const rund::AccelCheck projected =
      ProjectMetalView(pick, source, mode, views, view_binds, candidate);
  if (!projected.ok) {
    return projected;
  }
  if (candidate == nullptr) {
    return rund::AccelCheck{true, "ok"};
  }
  MetalAdapter *const adapter = MetalAdapterFromPick(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_metal_unavailable"};
  }
  bool needs_gather = false;
  bool needs_scatter = false;
  for (const MetalViewTransfer &transfer : candidate->transfers) {
    needs_gather = needs_gather || transfer.input;
    needs_scatter = needs_scatter || !transfer.input;
  }
  if (needs_gather) {
    candidate->gather_pipeline = AcquireMetalViewPipeline(
        *adapter, "pipeline.view.gather.u32", "rund_pipeline_view_gather");
  }
  if (needs_scatter) {
    candidate->scatter_pipeline = AcquireMetalViewPipeline(
        *adapter, "pipeline.view.scatter.u32", "rund_pipeline_view_scatter");
  }
  if ((needs_gather && candidate->gather_pipeline == nullptr) ||
      (needs_scatter && candidate->scatter_pipeline == nullptr)) {
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  out = std::move(candidate);
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
