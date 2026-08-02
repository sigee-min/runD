#pragma once

#include "../../kernel/backend/manifest.hpp"

namespace rund::kernel {
struct ComputePlan;
}

namespace rund::node::accel::detail {

struct BoundStep;
struct KernelExecutionStep;

[[nodiscard]] PreparedBackendManifest BuildVulkanBackendManifest(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const BoundStep *bound, std::uint64_t max_dispatch_groups) noexcept;

} // namespace rund::node::accel::detail
