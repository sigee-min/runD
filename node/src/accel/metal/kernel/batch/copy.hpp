#pragma once

#include "model.hpp"

#include <span>

namespace rund::node::accel::detail::metalbatch {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] bool pack(std::span<const BatchMapView> views,
                        const BatchMapPlan &plan, const Maps &maps,
                        const Workspace &workspace);
[[nodiscard]] bool unpack(std::span<const BatchMapView> views,
                          const BatchMapPlan &plan, const Maps &maps,
                          const Workspace &workspace);
#endif

} // namespace rund::node::accel::detail::metalbatch
