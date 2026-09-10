#pragma once

#include "../graph_resident.hpp"

#include "../../graph_wavefront.hpp"

#include <kernel/program/compute/lowering/model.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace rund::node::accel::detail {

[[nodiscard]] bool graph_resident_endpoints_valid(
    const DeviceVsmPageGeometry &, const DeviceVsmResidentSet &) noexcept;

[[nodiscard]] bool graph_resident_stages_valid(
    std::span<const DeviceVsmGraphResidentStageSource>,
    const DeviceVsmGraphResidentProof &, const char *&) noexcept;

namespace device_vsm_graph_resident_source::detail {

[[nodiscard]] const char *graph_resident_metal_type(
    const DeviceVsmGraphResidentProof &) noexcept;
[[nodiscard]] const char *graph_resident_vulkan_type(
    const DeviceVsmGraphResidentProof &) noexcept;
[[nodiscard]] const char *graph_resident_zero_literal(
    const DeviceVsmGraphResidentProof &) noexcept;

[[nodiscard]] rund::kernel::compute_lowering_detail::ParsedIR merged_helpers(
    std::span<const DeviceVsmGraphResidentStageSource>);

[[nodiscard]] const DeviceVsmGraphResidentResource *resource(
    const DeviceVsmGraphResidentProof &, std::uint32_t) noexcept;

[[nodiscard]] std::string offset(
    const DeviceVsmGraphResidentRegion &,
    const DeviceVsmGraphResidentRegion &, std::uint32_t,
    const std::string &, const std::string &);

[[nodiscard]] std::size_t external_count(
    const DeviceVsmGraphResidentProof &) noexcept;

[[nodiscard]] bool append_load(
    std::string &, const DeviceVsmGraphResidentProof &, std::uint32_t,
    const std::string &, const std::string &, std::uint32_t,
    const std::string &, const std::string &);
[[nodiscard]] bool append_store(
    std::string &, const DeviceVsmGraphResidentProof &, std::uint32_t,
    const std::string &, const std::string &, std::uint32_t,
    const std::string &, const std::string &);
[[nodiscard]] bool append_stage(
    std::string &, const DeviceVsmGraphResidentProof &,
    const DeviceVsmGraphResidentStage &,
    const DeviceVsmGraphResidentStageSource &, const rund::kernel::ArtifactKey &,
    const std::string &, const std::string &, std::uint32_t,
    const std::string &);

[[nodiscard]] bool append_metal_store(
    std::string &, const DeviceVsmGraphResidentProof &, std::uint32_t,
    const std::string &, const std::string &, std::uint32_t,
    const std::string &, const std::string &);
[[nodiscard]] bool append_metal_mapped_load(
    std::string &, const DeviceVsmGraphResidentProof &,
    const DeviceVsmGraphResidentResource &, std::uint32_t,
    const std::string &);
[[nodiscard]] bool append_metal_stage(
    std::string &, const DeviceVsmGraphResidentProof &,
    const DeviceVsmGraphResidentStage &,
    const DeviceVsmGraphResidentStageSource &, const rund::kernel::ArtifactKey &,
    const std::string &, const std::string &, const std::string &,
    std::uint32_t);

[[nodiscard]] bool graph_resident_key(
    std::span<const DeviceVsmGraphResidentStageSource>,
    const DeviceVsmGraphResidentProof &, const DeviceVsmGraphWavefrontProof &,
    rund::kernel::ArtifactKey &) noexcept;

[[nodiscard]] std::string vulkan_source(
    const rund::kernel::ArtifactKey &,
    std::span<const DeviceVsmGraphResidentStageSource>,
    const DeviceVsmGraphResidentProof &, const DeviceVsmGraphWavefrontProof &);

[[nodiscard]] std::string metal_source(
    const rund::kernel::ArtifactKey &,
    std::span<const DeviceVsmGraphResidentStageSource>,
    const DeviceVsmGraphResidentProof &, const DeviceVsmGraphWavefrontProof &);

} // namespace device_vsm_graph_resident_source::detail

} // namespace rund::node::accel::detail
