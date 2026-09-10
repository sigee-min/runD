#pragma once

#include "../persistent_product/local.hpp"

namespace rund_node_test_device_vsm_product {

[[nodiscard]] int CheckDeviceVsmProduct(
    rund::compute::Backend,
    rund_node_test_persistent_product::NativeQueueCounter) noexcept;
[[nodiscard]] bool
CheckDeviceVsmWarmReuse(rund::compute::Backend,
                        rund_node_test_persistent_product::NativeQueueCounter,
                        bool &) noexcept;

} // namespace rund_node_test_device_vsm_product
