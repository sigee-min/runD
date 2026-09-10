#pragma once

#include "../internal.hpp"

#include "src/compute/virtual/run/device_vsm/model.hpp"

namespace rund_node_test_virtual::product::graph_pointwise_depth_seven {

using Owner =
    rund::compute::detail::device_vsm_product_detail::DeviceVsmProductOwner;

[[nodiscard]] std::shared_ptr<Owner>
owner_of(const ResidentCase &test_case) noexcept;

} // namespace rund_node_test_virtual::product::graph_pointwise_depth_seven
