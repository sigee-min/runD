#pragma once

#include "../../context/internal/execution.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/admission.hpp>

namespace rund::node::accel::detail::step {

[[nodiscard]] MapSemantic BuildMapSemantic(
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input);

} // namespace rund::node::accel::detail::step
