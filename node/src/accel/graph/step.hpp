#pragma once

#include "compile.hpp"
#include <kernel/program/compute/artifact.hpp>

#include <span>

namespace rund::node::accel::detail {

[[nodiscard]] KernelExecutionStep BuildMapKernelExecutionStep(
    rund::kernel::LoweringArtifact artifact,
    rund::kernel::compute_lowering_detail::ComputeInputAdmission input,
    KernelBindingIndices binding_indices, std::uint64_t element_count);

[[nodiscard]] KernelExecutionStep
BuildKernelExecutionStep(GraphCompileNode &&node);

struct FrozenDispatchCount final {
  std::uint64_t count = 0u;
  bool ok = false;
  const char *reason = "compute_plan_invalid";
};

[[nodiscard]] FrozenDispatchCount
BuildMapDispatchCount(const rund::kernel::ExecutionMetadata &metadata,
                      std::uint64_t element_count,
                      const rund::kernel::ComputeCaps &caps,
                      std::uint64_t phase_id);

[[nodiscard]] FrozenDispatchCount
BuildOriginalDispatchCount(std::span<const GraphCompileNode> nodes,
                           const rund::kernel::ComputeCaps &caps,
                           std::uint64_t phase_offset = 0u);

} // namespace rund::node::accel::detail
