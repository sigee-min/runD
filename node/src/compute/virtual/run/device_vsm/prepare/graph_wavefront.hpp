#pragma once

#include "../../../../../accel/kernel/residency/device_vsm/proof.hpp"
#include "../../../../pipeline/residency/model.hpp"

#include <cstdint>
#include <span>

namespace rund::compute::detail::device_vsm_product_detail {

[[nodiscard]] bool project_graph_wavefront(
    const residency::TiledGraphPlan &, std::uint64_t active_page_count,
    std::uint32_t map_stage, std::uint32_t collective_stage,
    node::accel::detail::DeviceVsmGraphWavefrontProof &) noexcept;

[[nodiscard]] bool
project_graph_page_map(const residency::TiledGraphPlan &,
                       std::span<const std::uint32_t>,
                       node::accel::detail::DeviceVsmPageMap &) noexcept;

} // namespace rund::compute::detail::device_vsm_product_detail
