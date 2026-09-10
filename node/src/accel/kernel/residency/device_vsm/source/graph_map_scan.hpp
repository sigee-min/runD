#pragma once

#include "../proof.hpp"

#include "../../../../context/internal/execution.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/admission.hpp>

namespace rund::node::accel::detail {

struct DeviceVsmGraphMapScanArtifact final {
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::ComputePlan plan{};
  DeviceVsmScanProof proof{};
  const char *reason{"device_vsm_graph_scan_invalid"};

  [[nodiscard]] explicit operator bool() const noexcept {
    return artifact.ok && plan.ok && proof.workgroup_width != 0u &&
           proof.stage_count == 2u;
  }
};

[[nodiscard]] DeviceVsmGraphMapScanArtifact BuildDeviceVsmGraphMapScanArtifact(
    const rund::kernel::LoweringArtifact &,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &,
    const MapSemantic &, const rund::kernel::ScanPlan &,
    const DeviceVsmPageGeometry &) noexcept;

} // namespace rund::node::accel::detail
