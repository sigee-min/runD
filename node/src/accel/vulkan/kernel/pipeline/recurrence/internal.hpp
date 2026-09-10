#pragma once

#include "../recurrence.hpp"

#include "../../../../kernel/recurrence/plan.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] std::span<const std::uint64_t>
RecurrenceHistoryPitches(const MapRecurrence &recurrence) noexcept;

[[nodiscard]] bool ValidRecurrence(const MapRecurrence &recurrence) noexcept;

[[nodiscard]] bool RuntimeRecurrenceMatchesPreparedPlan(
    const MapRecurrencePreparationPlan &planned,
    const MapRecurrence &recurrence, std::uint64_t route_group_count,
    std::uint64_t route_history_group_count, std::uint64_t alignment) noexcept;

[[nodiscard]] bool RuntimeRecurrenceMatchesPlan(
    const BackendRun &owner, const MapRecurrence &recurrence,
    std::uint64_t route_group_count, std::uint64_t route_history_group_count,
    std::uint64_t alignment) noexcept;

[[nodiscard]] bool SameRuntimeRecurrenceTemplate(
    const BackendRun &left_owner, const MapRecurrence &left,
    const BackendRun &right_owner, const MapRecurrence &right,
    std::uint64_t alignment) noexcept;

[[nodiscard]] bool ValidRecurrenceReservation(
    const PreparedKernelTemplateRegistry &registry,
    const PreparedPipelineStatusLayout &status, std::uint64_t group_count,
    std::uint64_t history_group_count,
    std::uint64_t terminal_template_group_capacity,
    std::uint64_t history_template_group_capacity,
    std::uint64_t template_count) noexcept;

[[nodiscard]] rund::AccelCheck AcquireVulkanRecurrenceTemplate(
    PreparedKernelTemplateRegistry &registry, VulkanAdapter &adapter,
    const rund::AccelDevice &pick, const BackendRun &signature,
    const MapRecurrence &recurrence, std::uint64_t group_capacity,
    std::shared_ptr<VulkanMapRecurrenceTemplate> &prepared);

#endif

} // namespace rund::node::accel::detail
