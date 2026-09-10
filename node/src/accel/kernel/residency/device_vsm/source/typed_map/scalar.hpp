#pragma once

#include "internal.hpp"

#include <string>

namespace rund::node::accel::detail::device_vsm_typed_map {

[[nodiscard]] bool
    parameter_free_total_scalar_op_supported(rund::kernel::IrOp) noexcept;

[[nodiscard]] bool validate_parameter_free_total_u32_scalar(
    const rund::kernel::LoweringArtifact &,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission
        &) noexcept;

[[nodiscard]] bool append_metal_u32_scalar_function(
    std::string &, const rund::kernel::LoweringArtifact &,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &,
    const char *);

[[nodiscard]] bool append_vulkan_u32_scalar_function(
    std::string &, const rund::kernel::LoweringArtifact &,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &,
    const char *);

} // namespace rund::node::accel::detail::device_vsm_typed_map
