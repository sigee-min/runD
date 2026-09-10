#pragma once

#include "../graph_map_reduce/wavefront.hpp"
#include "../graph_pointwise.hpp"
#include "../typed_map/internal.hpp"
#include "../typed_map/scalar.hpp"

#include <cstdint>
#include <string>

namespace rund::node::accel::detail::device_vsm_graph_pointwise {

[[nodiscard]] bool validate_stage(
    const rund::kernel::LoweringArtifact &,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &,
    const MapSemantic &) noexcept;

[[nodiscard]] rund::kernel::compute_lowering_detail::ParsedIR
build_io_ir(std::size_t, rund::kernel::ComputeScalar,
            rund::kernel::ComputeDomain, std::uint32_t);

[[nodiscard]] std::string
metal_source(const rund::kernel::ArtifactKey &,
             std::span<const DeviceVsmGraphPointwiseStage>,
             const DeviceVsmGraphPointwiseTopology &,
             const DeviceVsmPageGeometry &, const DeviceVsmPageMap &,
             const DeviceVsmGraphWavefrontProof &);

[[nodiscard]] std::string
vulkan_source(const rund::kernel::ArtifactKey &,
              std::span<const DeviceVsmGraphPointwiseStage>,
              const DeviceVsmGraphPointwiseTopology &,
              const DeviceVsmPageGeometry &, const DeviceVsmPageMap &,
              const DeviceVsmGraphWavefrontProof &);

} // namespace rund::node::accel::detail::device_vsm_graph_pointwise
