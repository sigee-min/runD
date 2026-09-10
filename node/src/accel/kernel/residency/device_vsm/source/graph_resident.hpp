#pragma once

#include "graph_resident/model.hpp"

#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] DeviceVsmGraphResidentArtifact
BuildDeviceVsmGraphResidentArtifact(
    std::span<const DeviceVsmGraphResidentStageSource>,
    const DeviceVsmGraphResidentProof &, const DeviceVsmGraphWavefrontProof &,
    const DeviceVsmPageGeometry &, const DeviceVsmResidentSet &) noexcept;

} // namespace rund::node::accel::detail
