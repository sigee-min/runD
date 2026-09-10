#pragma once

#include "../../recurrence/plan.hpp"
#include "../model.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck plan_map_recurrence_reservation(
    const BackendOps &ops, const MapRecurrencePreparationPlan &plan,
    PreparedMapRecurrenceReservation &reservation) noexcept;

[[nodiscard]] bool scale_map_recurrence_route_reservation(
    const PreparedMapRecurrenceReservation &value, std::uint64_t copies,
    PreparedMapRecurrenceReservation &scaled) noexcept;

[[nodiscard]] PreparedMapRecurrenceReservation
map_recurrence_route_reservation(
    const PreparedMapRecurrenceReservation &value) noexcept;

[[nodiscard]] PreparedMapRecurrenceReservation
map_recurrence_template_reservation(
    const PreparedMapRecurrenceReservation &value) noexcept;

[[nodiscard]] bool same_map_recurrence_reservation(
    const PreparedMapRecurrenceReservation &left,
    const PreparedMapRecurrenceReservation &right) noexcept;

[[nodiscard]] bool accumulate_map_recurrence_template(
    PreparedMapRecurrenceReservation &target,
    const PreparedMapRecurrenceReservation &value) noexcept;

[[nodiscard]] bool accumulate_map_recurrence_route(
    PreparedMapRecurrenceReservation &target,
    const PreparedMapRecurrenceReservation &value) noexcept;

[[nodiscard]] rund::AccelCheck verify_map_recurrence_template_variants(
    const BackendOps &ops, const MapRecurrencePreparationPlan &plan,
    const PreparedMapRecurrenceReservation &combined,
    PreparedMapRecurrenceReservation &terminal,
    PreparedMapRecurrenceReservation &history) noexcept;

[[nodiscard]] bool project_map_recurrence_reservation(
    const PreparedMapRecurrenceReservation &recurrence,
    PreparedKernelPipelineReservation &projection) noexcept;

} // namespace rund::node::accel::detail
