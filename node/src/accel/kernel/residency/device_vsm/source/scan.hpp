#pragma once

#include "../proof.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/scan/model.hpp>

namespace rund::node::accel::detail {

struct DeviceVsmScanArtifact final {
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::ComputePlan plan{};
  DeviceVsmScanProof proof{};
  const char *reason{"device_vsm_scan_invalid"};

  [[nodiscard]] explicit operator bool() const noexcept {
    return artifact.ok && plan.ok && proof.semantic.ok &&
           proof.workgroup_width == 256u;
  }
};

[[nodiscard]] DeviceVsmScanArtifact BuildDeviceVsmScanArtifact(
    const rund::kernel::ScanPlan &, rund::kernel::ComputeApi,
    const DeviceVsmPageGeometry &, DeviceVsmScanMap = {}) noexcept;

} // namespace rund::node::accel::detail
