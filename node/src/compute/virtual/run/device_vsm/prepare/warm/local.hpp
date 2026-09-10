#pragma once

#include "../../internal.hpp"

namespace rund::compute::detail::device_vsm_product_detail::warm_detail {

[[nodiscard]] std::uint64_t page_count(const VirtualRunProjection &) noexcept;
[[nodiscard]] bool owner_ready(const DeviceVsmProductOwner *) noexcept;
[[nodiscard]] bool
owner_shape_matches(const VirtualPipelineState &, const VirtualRunProjection &,
                    const node::accel::detail::DeviceVsmIdentity &,
                    const VirtualDeviceVsmRouteProof &,
                    const DeviceVsmProductOwner &) noexcept;
[[nodiscard]] bool pipelines_match(VirtualPipelineState &,
                                   const VirtualRunProjection &,
                                   const DeviceVsmProductOwner &) noexcept;
[[nodiscard]] bool
backend_bindings_match(const VirtualPipelineState &,
                       const VirtualRunProjection &, const AccelDeviceState *,
                       const DeviceVsmProductOwner &) noexcept;
[[nodiscard]] bool same_wavefront(
    const node::accel::detail::DeviceVsmGraphWavefrontProof &,
    const node::accel::detail::DeviceVsmGraphWavefrontProof &) noexcept;
[[nodiscard]] bool validate_pointwise(const VirtualPipelineState &,
                                      const VirtualRunProjection &,
                                      const DeviceVsmProductOwner &) noexcept;
[[nodiscard]] bool validate_resident(const VirtualPipelineState &,
                                     const VirtualRunProjection &,
                                     const DeviceVsmProductOwner &,
                                     const char *&reason) noexcept;

} // namespace rund::compute::detail::device_vsm_product_detail::warm_detail
