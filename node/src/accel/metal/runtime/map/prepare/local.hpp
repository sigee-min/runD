#pragma once

#include "../api.hpp"
#include "../control.hpp"

namespace rund::node::accel::detail::metal_map_prepare {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool same_bindings(const MetalMapTemplateResources &,
                                 const rund::kernel::BindingSet &) noexcept;
[[nodiscard]] rund::AccelCheck
prepare_route_resources(MetalAdapter &, const rund::AccelDevice &,
                        const rund::kernel::ComputePlan &,
                        const rund::kernel::ComputeDispatchWindow *,
                        rund::kernel::u64 window_count,
                        const rund::kernel::BindingSet &, const BoundControl &,
                        std::shared_ptr<const MetalMapTemplateResources>,
                        std::shared_ptr<void> &, std::uint32_t iterations);
#endif

[[nodiscard]] rund::AccelCheck
prepare_template(const rund::AccelDevice &, const rund::kernel::ComputePlan &,
                 const rund::kernel::LoweringArtifact &,
                 rund::kernel::LoweringArtifact *owned_artifact,
                 const rund::kernel::ComputeDispatchWindow *,
                 rund::kernel::u64 window_count,
                 const rund::kernel::BindingSet &, const BoundControl &,
                 std::shared_ptr<const MetalMapTemplateResources> &);

} // namespace rund::node::accel::detail::metal_map_prepare
