#pragma once

#include "../internal.hpp"

#include <cstdint>

namespace rund::node::accel::detail::metal_device_vsm::prepare {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

// Per-call facts derived from the authenticated proof.  Retained native
// state remains exclusively in metal_device_vsm::Owner.
struct Shape final {
  bool graph{};
  bool graph_map{};
  bool graph_pointwise_map{};
  bool window_ring{};
  bool ring{};
  bool ring_storage{};
  DeviceVsmWindowRingPlan window_ring_plan{};
};

// Native byte counts are derived once during allocation and reused by the
// final materialization phase.  This is transient preparation scratch, not a
// second owner or capability record.
struct ResourcePlan final {
  std::uint64_t parameter_bytes{};
  std::uint64_t ring_state_bytes{};
  std::uint64_t ring_region_bytes{};
};

[[nodiscard]] rund::AccelCheck
ValidateShapeAndBindings(Owner &, const rund::AccelDevice &,
                         const DeviceVsmProof &, Shape &) noexcept;

[[nodiscard]] rund::AccelCheck AllocateResources(Owner &, MetalAdapter &,
                                                 const DeviceVsmProof &,
                                                 const Shape &,
                                                 ResourcePlan &) noexcept;

[[nodiscard]] bool RecordGraph(Owner &, MetalAdapter &) noexcept;

#endif

} // namespace rund::node::accel::detail::metal_device_vsm::prepare
