#pragma once

#include "../actual.hpp"
#include "../local.hpp"
#include "../wait.hpp"

#include "src/compute/device/state.hpp"

#include <memory>

namespace rund_node_test_pipeline_residency::device_vsm_test {

[[nodiscard]] rund::kernel::ComputeApi ApiFor(rund::compute::Backend) noexcept;
[[nodiscard]] rund::compute::Target TargetFor(rund::compute::Backend) noexcept;
[[nodiscard]] std::shared_ptr<accel::DeviceVsmProof>
BuildProof(rund::kernel::ComputeApi, std::uint64_t, const accel::UploadRoute &,
           const accel::UploadRoute &);
[[nodiscard]] wait_detail::Result
CloseAuthority(wait_detail::Owner &, const accel::DeviceVsmFinal &) noexcept;
[[nodiscard]] bool
RunActual(rund::compute::Backend,
          const std::shared_ptr<rund::compute::detail::DeviceState> &,
          std::uint64_t, ActualPrepare, ActualQueueCount, std::uint64_t &);

} // namespace rund_node_test_pipeline_residency::device_vsm_test
