#pragma once

#include "actual.hpp"

#include <memory>

namespace rund::compute::detail {
struct DeviceState;
}

namespace rund_node_test_pipeline_residency::device_vsm_test {

[[nodiscard]] bool CheckActualBinaryDeviceVsm(
    rund::compute::Backend,
    const std::shared_ptr<rund::compute::detail::DeviceState> &, ActualPrepare,
    ActualQueueCount) noexcept;

} // namespace rund_node_test_pipeline_residency::device_vsm_test
