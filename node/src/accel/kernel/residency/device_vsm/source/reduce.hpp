#pragma once

#include "../proof.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/reduce/model.hpp>

namespace rund::node::accel::detail {

struct DeviceVsmReduceArtifact final {
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::ComputePlan plan{};
  DeviceVsmReduceProof proof{};
  const char *reason{"device_vsm_reduce_invalid"};

  [[nodiscard]] explicit operator bool() const noexcept {
    return artifact.ok && plan.ok && proof.semantic.ok &&
           proof.workgroup_width == 256u;
  }
};

[[nodiscard]] DeviceVsmReduceArtifact
BuildDeviceVsmReduceArtifact(const rund::kernel::ReducePlan &,
                             rund::kernel::ComputeApi,
                             const DeviceVsmPageGeometry &) noexcept;

} // namespace rund::node::accel::detail
