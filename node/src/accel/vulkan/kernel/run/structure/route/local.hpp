#pragma once

#include "../../route.hpp"

#include "../../../../../kernel/backend/template/model.hpp"
#include "../../../../../kernel/backend/template/reservation.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
struct VulkanAdapter;

[[nodiscard]] rund::AccelCheck PlanVulkanStepStructure(
    const KernelExecutionStep &step, const rund::kernel::ComputePlan &plan,
    const BoundStep *bound, const KernelViewLayout *views,
    std::uint64_t max_dispatch_groups,
    PreparedKernelRouteReservation &reservation) noexcept;
[[nodiscard]] rund::AccelCheck CompleteVulkanRouteCaptureStructure(
    const KernelViewLayout *views, std::uint64_t reset_count,
    const VulkanAdapter &adapter,
    PreparedKernelRouteReservation &reservation) noexcept;
#endif

} // namespace rund::node::accel::detail
