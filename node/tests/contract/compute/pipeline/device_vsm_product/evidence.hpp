#pragma once

#include "../persistent_product/model.hpp"
#include "route.hpp"

#include "src/accel/kernel/residency/device_vsm/geometry.hpp"
#include "src/compute/virtual/run/device_vsm/model.hpp"

#include <rund/compute/stats.hpp>

namespace rund_node_test_device_vsm_product {

[[nodiscard]] bool ExactDeviceVsmEvidence(const RouteObservation &,
                                          std::uint64_t, std::uint64_t,
                                          std::uint64_t, std::uint64_t,
                                          std::uint64_t) noexcept;

[[nodiscard]] bool ExactWholeRunTransfers(
    const rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner &,
    const rund::compute::Stats &, std::uint64_t) noexcept;

[[nodiscard]] bool ExactWindowRingEvidence(
    const RouteObservation &, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t,
    const rund::node::accel::detail::DeviceVsmWindowRingPlan &) noexcept;

[[nodiscard]] bool
ExactPublication(rund_node_test_persistent_product::PublicationSnapshot,
                 rund_node_test_persistent_product::PublicationSnapshot,
                 rund_node_test_persistent_product::PublicationSnapshot,
                 rund_node_test_persistent_product::PublicationSnapshot,
                 std::uint64_t, std::uint64_t, std::uint64_t) noexcept;

} // namespace rund_node_test_device_vsm_product
