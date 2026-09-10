#pragma once

#include "../internal.hpp"

// Preparation helpers are deliberately exposed as declarations only.  The
// authenticated proof, Owner model, descriptor formula, and graph binding
// identity helpers remain owned by device_vsm/internal.hpp.
namespace rund::node::accel::detail::vulkan_device_vsm::prepare {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] rund::AccelCheck ValidateShape(
    const DeviceVsmProof &, bool &ring, bool &window_ring, bool &ring_storage,
    bool &graph, bool &graph_map, bool &graph_pointwise_map,
    DeviceVsmWindowRingPlan &window_ring_plan, std::uint64_t &ring_state_bytes,
    std::uint64_t &ring_scratch_bytes,
    std::uint64_t &ring_region_bytes) noexcept;

[[nodiscard]] rund::AccelCheck BindResidents(const rund::AccelDevice &,
                                             const DeviceVsmProof &,
                                             Owner &) noexcept;

[[nodiscard]] rund::AccelCheck PrepareGraph(const rund::AccelDevice &,
                                            const DeviceVsmProof &,
                                            Owner &) noexcept;

[[nodiscard]] bool FillGraphBindings(Owner &) noexcept;

[[nodiscard]] rund::AccelCheck RecordGraph(Owner &) noexcept;

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm::prepare
