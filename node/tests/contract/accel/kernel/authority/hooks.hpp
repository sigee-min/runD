#pragma once

#include <cstdint>
#include <span>

namespace rund::kernel {

struct ResidentBufferRef;

} // namespace rund::kernel

namespace rund::node::accel::detail {

struct BackendRun;
struct BackendTemplateRouteDemand;
struct BackendWindow;
struct PreparedKernelPipelineReservation;
struct PreparedKernelPublicationViewIdentity;
struct PreparedKernelRouteReservation;
struct PreparedKernelRun;
struct PreparedMapRecurrenceReservation;
struct PreparedPipelineFailure;

[[nodiscard]] bool AccumulatePreparedKernelRouteProjectionForContract(
    PreparedKernelPipelineReservation &projection,
    const PreparedKernelRouteReservation &route,
    std::uint64_t compact_entry_count, std::uint64_t occurrence_count,
    std::uint64_t window_count) noexcept;
[[nodiscard]] bool BackendTemplateRouteDemandForContract(
    std::uint64_t owner_count, std::uint64_t route_copies,
    BackendTemplateRouteDemand &demand) noexcept;
[[nodiscard]] bool PlanBackendTemplateRouteDemandsForContract(
    std::span<const PreparedKernelRun *const> runs, std::uint32_t route_copies,
    std::span<BackendTemplateRouteDemand> demands,
    std::uint64_t &unique_route_count, std::uint64_t &template_count) noexcept;
[[nodiscard]] bool BackendPreparationCursorLifecycleForContract(
    BackendRun &run, BackendTemplateRouteDemand demand) noexcept;
[[nodiscard]] bool ScalePreparedMapRecurrenceRoutesForContract(
    const PreparedMapRecurrenceReservation &reservation, std::uint64_t copies,
    PreparedMapRecurrenceReservation &scaled) noexcept;
[[nodiscard]] bool MaterializePreparedPipelineOccurrenceForContract(
    const BackendWindow &template_window, std::uint32_t outer,
    std::uint32_t outer_bound, std::uint32_t inner, std::uint32_t inner_bound,
    std::uint32_t route, std::uint32_t inner_advance, bool transduced,
    PreparedPipelineFailure &failure) noexcept;
[[nodiscard]] bool PreparedPublicationViewMatchesForContract(
    const rund::kernel::ResidentBufferRef &view,
    const PreparedKernelPublicationViewIdentity &identity) noexcept;

} // namespace rund::node::accel::detail
