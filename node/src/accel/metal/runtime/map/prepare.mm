#include "api.hpp"

#include <utility>

namespace rund::node::accel::detail {

rund::AccelCheck PrepareMetalMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<void> &resources, const std::uint32_t iterations) {
  std::shared_ptr<const MetalMapTemplateResources> prepared;
  const rund::AccelCheck template_ready = PrepareMetalMapTemplate(
      pick, plan, artifact, windows, window_count, bindings, control, prepared);
  return template_ready.ok
             ? PrepareMetalMapRoute(pick, plan, artifact, windows, window_count,
                                    bindings, control, std::move(prepared),
                                    resources, iterations)
             : template_ready;
}

} // namespace rund::node::accel::detail
