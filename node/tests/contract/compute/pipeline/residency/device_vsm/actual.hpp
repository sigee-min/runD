#pragma once

#include "src/accel/kernel/residency/device_vsm.hpp"

#include <accel/device.hpp>
#include <rund/compute/backend.hpp>

#include <cstdint>

namespace rund_node_test_pipeline_residency::device_vsm_test {

using ActualPrepare = rund::node::accel::detail::DeviceVsmPreparation (*)(
    const rund::AccelDevice &,
    const std::shared_ptr<const rund::node::accel::detail::DeviceVsmProof>
        &) noexcept;
using ActualQueueCount = bool (*)(const rund::AccelDevice &,
                                  std::uint64_t &) noexcept;

[[nodiscard]] bool CheckActualDeviceVsm(rund::compute::Backend, ActualPrepare,
                                        ActualQueueCount) noexcept;

} // namespace rund_node_test_pipeline_residency::device_vsm_test
