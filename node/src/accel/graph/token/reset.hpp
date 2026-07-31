#pragma once

#include "../../context/internal/execution.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

[[nodiscard]] bool
PlanResets(std::span<const KernelExecutionStep> steps,
           std::span<const rund::kernel::BufferRole> roles,
           std::span<const rund::GraphBufferVisibility> visibilities,
           std::span<const std::uint64_t> aliases,
           std::span<const SourceStep> sources,
           std::span<const std::uint64_t> reset_bindings,
           std::vector<ResetPlan> &plans) noexcept;

} // namespace rund::node::accel::detail
