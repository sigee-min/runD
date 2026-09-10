#pragma once

#include "../proof.hpp"

#include "../../../../context/internal/execution.hpp"

#include <kernel/program/compute/lowering/admission.hpp>

#include <span>

namespace rund::node::accel::detail {

struct DeviceVsmGraphPointwiseArtifact final {
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::ComputePlan plan{};
  DeviceVsmGraphPointwiseProof proof{};
  const char *reason{"device_vsm_graph_pointwise_invalid"};

  [[nodiscard]] explicit operator bool() const noexcept {
    return artifact.ok && plan.ok && proof.workgroup_width != 0u;
  }
};

struct DeviceVsmGraphPointwiseStage final {
  const rund::kernel::LoweringArtifact *artifact{};
  const rund::kernel::compute_lowering_detail::ComputeInputAdmission *input{};
  const MapSemantic *semantic{};
};

[[nodiscard]] DeviceVsmGraphPointwiseArtifact
BuildDeviceVsmGraphPointwiseArtifact(
    std::span<const DeviceVsmGraphPointwiseStage>,
    const DeviceVsmGraphPointwiseTopology &, const DeviceVsmPageGeometry &,
    const DeviceVsmPageMap &, const DeviceVsmGraphWavefrontProof &) noexcept;

[[nodiscard]] inline DeviceVsmGraphPointwiseArtifact
BuildDeviceVsmGraphPointwiseArtifact(
    const std::span<const DeviceVsmGraphPointwiseStage> stages,
    const DeviceVsmGraphPointwiseTopology &topology,
    const DeviceVsmPageGeometry &geometry,
    const DeviceVsmGraphWavefrontProof &wavefront) noexcept {
  return BuildDeviceVsmGraphPointwiseArtifact(stages, topology, geometry,
                                              DeviceVsmPageMap{}, wavefront);
}

} // namespace rund::node::accel::detail
