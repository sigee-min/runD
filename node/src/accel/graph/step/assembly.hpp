#pragma once

#include "../step.hpp"

#include <cstdint>

namespace rund::node::accel::detail::step {

[[nodiscard]] KernelExecutionStep AssembleKernelExecutionStep(
    rund::kernel::LoweringArtifact artifact,
    rund::kernel::compute_lowering_detail::ComputeInputAdmission input,
    KernelBindingIndices binding_indices, Operation operation,
    std::uint64_t primitive_hash_hi, std::uint64_t primitive_hash_lo,
    std::uint64_t element_count, rund::kernel::GraphControl control);

} // namespace rund::node::accel::detail::step
