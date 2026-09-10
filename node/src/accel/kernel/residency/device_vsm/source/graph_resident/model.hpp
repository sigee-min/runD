#pragma once

#include "../../proof.hpp"
#include "../graph_pointwise/internal.hpp"

#include <kernel/program/compute/artifact.hpp>

#include <memory>

namespace rund::node::accel::detail {

struct DeviceVsmGraphResidentStageSource final {
  const rund::kernel::LoweringArtifact *artifact{};
  const rund::kernel::compute_lowering_detail::ComputeInputAdmission *input{};
  const MapSemantic *semantic{};
};

struct DeviceVsmGraphResidentArtifact final {
  rund::kernel::LoweringArtifact artifact{};
  rund::kernel::ComputePlan plan{};
  const char *reason{"device_vsm_graph_resident_invalid"};

  [[nodiscard]] explicit operator bool() const noexcept {
    return artifact.ok && plan.ok;
  }
};

} // namespace rund::node::accel::detail
