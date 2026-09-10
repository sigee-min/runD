#pragma once

#include "../../../../range_aggregate/model/plan.hpp"
#include "../proof.hpp"

#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/window/model.hpp>

#include <array>
#include <cstddef>

namespace rund::node::accel::detail {

class RangeExec;

struct DeviceVsmWindowMapSource final {
  const rund::kernel::LoweringArtifact *artifact{};
  const rund::kernel::compute_lowering_detail::ComputeInputAdmission *input{};
};

struct DeviceVsmWindowMapSources final {
  DeviceVsmWindowMapSource before{};
  DeviceVsmWindowMapSource before_second{};
  DeviceVsmWindowMapSource before_third{};
  DeviceVsmWindowMapSource after{};
  DeviceVsmWindowMapSource after_second{};
  DeviceVsmWindowMapSource after_third{};
};

struct DeviceVsmWindowArtifact final {
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::ComputePlan plan{};
  DeviceVsmWindowProof proof{};
  std::array<std::byte, 64u> parameters{};
  rund::kernel::ComputeDispatchWindow window{};
  bool ok{};
  const char *reason{"device_vsm_window_invalid"};

  [[nodiscard]] explicit operator bool() const noexcept { return ok; }
};

[[nodiscard]] bool DeviceVsmWindowRangeSupported(const RangeExec &) noexcept;
[[nodiscard]] bool DeviceVsmWindowRangeMutatesInput(const RangeExec &) noexcept;

[[nodiscard]] DeviceVsmWindowArtifact BuildDeviceVsmWindowArtifact(
    const RangePlan &, const rund::kernel::WindowPlan &,
    const DeviceVsmPageGeometry &, DeviceVsmWindowFusion = {},
    DeviceVsmWindowRingPlan = {},
    DeviceVsmWindowMapSources = {}) noexcept;

} // namespace rund::node::accel::detail
