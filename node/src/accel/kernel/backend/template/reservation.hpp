#pragma once

#include "model.hpp"

namespace rund::node::accel::detail {

struct BackendRun;
struct KernelExecution;
struct PreparedKernelProgramRoute;
struct PreparedKernelRouteReservation;

} // namespace rund::node::accel::detail

namespace rund::node::accel::detail::backend_template_plan {

[[nodiscard]] rund::AccelCheck
plan(const BackendRun &run, BackendShape shape,
     PreparedKernelRouteReservation &reservation) noexcept;

[[nodiscard]] rund::AccelCheck
plan_program(const KernelExecution &execution,
             const PreparedKernelProgramRoute &route, BackendShape shape,
             PreparedKernelRouteReservation &reservation) noexcept;

} // namespace rund::node::accel::detail::backend_template_plan
