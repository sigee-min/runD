#pragma once

#include "compile.hpp"
#include <kernel/program/compute/fusion.hpp>
#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <vector>

namespace rund::node::accel::detail {

struct FinalGraphSteps {
  std::vector<KernelExecutionStep> steps{};
  std::vector<std::uint8_t> barriers{};
  std::uint64_t removed_dispatch_count = 0u;
  std::uint64_t fused_operation_count = 0u;
  std::uint64_t fusion_rejection_count = 0u;
  const char *fusion_reason = "compute_fusion_invalid";
  bool ok = false;
  const char *reason = "accel_kernel_graph_invalid";
};

[[nodiscard]] rund::kernel::FusionPolicy FusionPolicyFor(
    const std::vector<rund::kernel::FusionNodePolicy> &nodes) noexcept;

[[nodiscard]] FinalGraphSteps BuildFinalGraphSteps(
    const rund::kernel::Graph &graph, const rund::kernel::FusionPlan &fusion,
    GraphCompileState &state, const rund::kernel::ComputeCaps &caps);

} // namespace rund::node::accel::detail
