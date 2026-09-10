#pragma once

#include "../../../../../accel/kernel/residency/device_vsm/proof.hpp"
#include "../../../../pipeline/residency/model.hpp"

#include <cstdint>
#include <span>

namespace rund::compute::detail::device_vsm_product_detail {

[[nodiscard]] bool project_graph_pointwise_topology(
    const residency::TiledGraphPlan &,
    std::span<const std::uint32_t> external_inputs,
    node::accel::detail::DeviceVsmGraphPointwiseTopology &) noexcept;

} // namespace rund::compute::detail::device_vsm_product_detail
