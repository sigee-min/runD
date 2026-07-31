#pragma once

#include "../fusion.hpp"
#include "../step.hpp"
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

struct FusedStepBindings {
  KernelBindingIndices indices{};
  std::size_t metadata_index = 0u;
  bool ok = false;
};

[[nodiscard]] FinalGraphSteps
RejectFinalGraphSteps(const char *reason,
                      const rund::kernel::FusionPlan &fusion) noexcept;
[[nodiscard]] bool FusedStepBindingsMatchMetadata(
    const FusedStepBindings &bindings,
    const rund::kernel::ExecutionMetadata &metadata) noexcept;
[[nodiscard]] FusedStepBindings
FusedStepBindingsFor(const rund::kernel::Graph &graph,
                     std::span<const GraphCompileNode> nodes,
                     const rund::kernel::ExecutionMetadata &metadata);

} // namespace rund::node::accel::detail
