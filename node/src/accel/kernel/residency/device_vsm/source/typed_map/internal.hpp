#pragma once

#include "../../../../../context/internal/execution.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/admission.hpp>
#include <kernel/program/compute/lowering/layout.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace rund::node::accel::detail::device_vsm_typed_map {

[[nodiscard]] bool validate_total_scalar(
    const rund::kernel::LoweringArtifact &,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &,
    const MapSemantic &, rund::kernel::ComputeScalar,
    rund::kernel::ComputeDomain, std::uint32_t) noexcept;

[[nodiscard]] bool validate_total_u64(
    const rund::kernel::LoweringArtifact &,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &,
    const MapSemantic &) noexcept;

[[nodiscard]] bool append_metal_header(
    std::string &, const rund::kernel::ArtifactKey &,
    const rund::kernel::compute_lowering_detail::ParsedIR &,
    std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &,
    bool emit_param_helpers = true);
[[nodiscard]] bool append_metal_value(
    std::string &, const rund::kernel::ArtifactKey &,
    const rund::kernel::compute_lowering_detail::ParsedIR &,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &,
    std::string &);
void append_metal_store(
    std::string &, const rund::kernel::compute_lowering_detail::ParsedIR &,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &,
    const std::string &, const std::string &,
    rund::kernel::ComputeScalar = rund::kernel::ComputeScalar::Lane64);

[[nodiscard]] bool append_vulkan_prelude(
    std::string &, const rund::kernel::ArtifactKey &,
    const rund::kernel::compute_lowering_detail::ParsedIR &,
    std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &,
    bool emit_param_helpers = true);
void append_vulkan_main(std::string &);
[[nodiscard]] bool append_vulkan_value(
    std::string &, const rund::kernel::ArtifactKey &,
    const rund::kernel::compute_lowering_detail::ParsedIR &,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &,
    std::string &);
void append_vulkan_store(
    std::string &, const rund::kernel::compute_lowering_detail::ParsedIR &,
    const std::vector<rund::kernel::compute_lowering_detail::BindingLayout> &,
    const std::string &, const std::string &,
    rund::kernel::ComputeScalar = rund::kernel::ComputeScalar::Lane64);

[[nodiscard]] std::string vulkan_scratch_value(rund::kernel::ComputeScalar,
                                               const std::string &);
[[nodiscard]] std::string metal_scratch_value(rund::kernel::ComputeScalar,
                                              const std::string &);

} // namespace rund::node::accel::detail::device_vsm_typed_map
