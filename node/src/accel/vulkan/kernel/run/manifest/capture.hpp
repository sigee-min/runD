#pragma once

#include "../../manifest.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool PlanVulkanCaptureManifest(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const BoundStep *bound, std::uint64_t max_dispatch_groups,
    PreparedBackendManifest &manifest) noexcept;

} // namespace rund::node::accel::detail
