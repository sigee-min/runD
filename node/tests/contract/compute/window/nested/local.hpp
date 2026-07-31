#pragma once

#include <rund/compute/backend.hpp>
#include <rund/compute/device.hpp>

namespace rund::node::test_contract::window {

[[nodiscard]] int CheckNestedWindow(rund::compute::Device &device,
                                    rund::compute::Backend backend);

} // namespace rund::node::test_contract::window
