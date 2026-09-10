#pragma once

#include "../local.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace rund::node::accel::detail::schedule {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

inline constexpr std::uint64_t ScheduleTimeoutNs = 30'000'000'000u;

[[nodiscard]] rund::AccelCheck Invalid() noexcept;
[[nodiscard]] rund::AccelCheck Unavailable() noexcept;

[[nodiscard]] bool UniqueLocals(const BackendResidencyScheduleRole &role,
                                std::size_t count) noexcept;

[[nodiscard]] bool ControlGeneration(
    const BackendResidencyScheduleRole &role, std::uint64_t epoch,
    std::uint32_t &generation) noexcept;

[[nodiscard]] VulkanTimelinePoint Point(const VulkanResidencyScheduleRun &run,
                                        std::uint64_t epoch) noexcept;

[[nodiscard]] VulkanResidencyScheduleRole &NativeRole(
    VulkanResidencyScheduleRun &run, std::uint64_t epoch) noexcept;

[[nodiscard]] const BackendResidencyScheduleRole &SourceRole(
    const VulkanResidencyScheduleRun &run, std::uint64_t epoch) noexcept;

[[nodiscard]] std::size_t LocalCount(const VulkanResidencyScheduleRun &run,
                                     std::uint64_t epoch) noexcept;

[[nodiscard]] bool Materialize(
    const BackendResidencyScheduleRequest &request,
    std::array<VulkanResidencyScheduleRole, ResidencyScheduleRoleCapacity>
        &roles,
    VulkanResidencyScheduleRole &tail, VulkanAdapter *&adapter,
    VulkanResidencyScheduleRun *&run) noexcept;

void ClearActive(VulkanResidencyScheduleRun &run) noexcept;
void Service(void *raw) noexcept;

#endif

} // namespace rund::node::accel::detail::schedule
