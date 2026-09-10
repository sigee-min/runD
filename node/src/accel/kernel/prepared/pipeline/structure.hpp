#pragma once

#include "../run.hpp"

#include "../model.hpp"
#include "../../recurrence.hpp"
#include "../../recurrence/plan.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

struct MapRecurrenceTemplateCapacities final {
  std::uint64_t terminal{};
  std::uint64_t history{};
};

struct RuntimeRecurrenceRoutePlan final {
  std::uint64_t entry_count{};
  std::uint64_t occurrence_count{};
  std::uint64_t window_count{};
  std::uint64_t nested_group_count{};
  std::uint64_t group_count{};
  std::uint64_t history_group_count{};
  std::uint64_t fingerprint_hi{};
  std::uint64_t fingerprint_lo{};
  MapRecurrenceTemplateCapacities template_capacities{};
  bool first_owner{};
};

void fingerprint_mix(std::uint64_t &hash, std::uint64_t value) noexcept;

void fingerprint_pipeline_header(std::uint64_t &hi, std::uint64_t &lo,
                                 const rund::AccelContext &context,
                                 PreparedKernelPipelineShape shape,
                                 std::uint64_t route_count) noexcept;

[[nodiscard]] PreparedKernelPipelineShape runtime_pipeline_shape(
    std::span<const BackendPublish> publications,
    std::span<const BackendRecurrence> recurrences,
    std::uint32_t declared_step_count, std::uint32_t route_copies,
    bool profile_steps) noexcept;

[[nodiscard]] bool publication_view_matches(
    const rund::kernel::ResidentBufferRef &view,
    const PreparedKernelPublicationViewIdentity &identity) noexcept;

[[nodiscard]] bool
valid_publication_shape(const PreparedKernelPipelineShape &shape) noexcept;

void fingerprint_route(
    std::uint64_t &hi, std::uint64_t &lo, std::uint64_t kernel_id,
    std::uint64_t graph_id_hi, std::uint64_t graph_id_lo,
    std::uint64_t node_count, rund::AccelApi api, std::uint64_t tile_count,
    const KernelViewLayout *views, const KernelScratchLayout *scratch,
    std::uint64_t entry_count, std::uint64_t occurrence_count,
    std::uint64_t window_count, std::uint64_t nested_group_count,
    std::uint64_t map_recurrence_group_count,
    std::uint64_t map_recurrence_history_group_count,
    std::uint64_t recurrence_hi, std::uint64_t recurrence_lo,
    std::uint32_t route_copies) noexcept;

[[nodiscard]] bool route_recurrence_shape(
    std::span<const PreparedKernelRun *const> runs,
    std::span<const BackendRecurrence> recurrences, const void *route_owner,
    std::uint64_t &entry_count, std::uint64_t &occurrence_count,
    std::uint64_t &window_count, std::uint64_t &nested_group_count,
    std::uint64_t &map_recurrence_group_count,
    std::uint64_t &map_recurrence_history_group_count,
    std::uint64_t &recurrence_hi, std::uint64_t &recurrence_lo) noexcept;

[[nodiscard]] bool plan_runtime_recurrence_routes(
    std::span<const PreparedKernelRun *const> runs,
    std::span<const BackendRecurrence> recurrences, std::uint32_t route_copies,
    std::span<RuntimeRecurrenceRoutePlan> plans) noexcept;

[[nodiscard]] bool same_program_recurrence_authority(
    const PreparedKernelProgramRoute &left_route,
    const PreparedKernelProgramRoute &right_route) noexcept;

[[nodiscard]] bool program_recurrence_template_capacities(
    const KernelExecution &execution,
    std::span<const PreparedKernelProgramRoute> routes, std::size_t route_index,
    const MapRecurrencePreparationPlan &route_plan,
    MapRecurrenceTemplateCapacities &capacities) noexcept;

[[nodiscard]] PreparedKernelPipelineReservation
plan_pipeline_structure_counts(std::uint64_t authored_entry_count,
                               std::uint64_t occurrence_count,
                               std::uint64_t window_count,
                               std::uint64_t nested_group_count) noexcept;

[[nodiscard]] PreparedKernelPipelineReservation
plan_pipeline_structure(std::span<const BackendRecurrence> recurrences) noexcept;

[[nodiscard]] bool accumulate_route_projection(
    PreparedKernelPipelineReservation &projection,
    const PreparedKernelRouteReservation &route,
    std::uint64_t compact_entry_count, std::uint64_t occurrence_count,
    std::uint64_t window_count) noexcept;

} // namespace rund::node::accel::detail
