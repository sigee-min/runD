#pragma once

#include "local.hpp"

#include "src/accel/kernel/residency/device_vsm/proof.hpp"

namespace rund_node_test_device_vsm_product::window_test {

[[nodiscard]] bool
ExactWindowFusion(const rund::node::accel::detail::DeviceVsmProof &,
                  PipelineShape) noexcept;

} // namespace rund_node_test_device_vsm_product::window_test
