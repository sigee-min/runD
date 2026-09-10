#pragma once

#include "../proof.hpp"

#include "../../../../context/internal/execution.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/admission.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

struct DeviceVsmGraphMapReduceArtifact final {
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::ComputePlan plan{};
  DeviceVsmGraphMapReduceProof proof{};
  const char *reason{"device_vsm_graph_invalid"};

  [[nodiscard]] explicit operator bool() const noexcept {
    return artifact.ok && plan.ok && proof.workgroup_width != 0u &&
           proof.wavefront.stage_count >= 2u;
  }
};

[[nodiscard]] DeviceVsmGraphMapReduceArtifact
BuildDeviceVsmGraphMapReduceArtifact(
    const rund::kernel::LoweringArtifact &,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &,
    const MapSemantic &, const rund::kernel::ReducePlan &,
    const DeviceVsmPageGeometry &,
    const DeviceVsmGraphWavefrontProof &) noexcept;

} // namespace rund::node::accel::detail
