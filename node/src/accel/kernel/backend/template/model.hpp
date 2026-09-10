#pragma once

#include <accel/check.hpp>

#include "../../view.hpp"

#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

struct BoundStep;
struct KernelExecutionStep;
struct PreparedKernelRouteReservation;

} // namespace rund::node::accel::detail

namespace rund::kernel {

struct ComputePlan;

} // namespace rund::kernel

namespace rund::node::accel::detail::backend_template_plan {

using StepStructurePlan = rund::AccelCheck (*)(
    const KernelExecutionStep &, const rund::kernel::ComputePlan &,
    const BoundStep *, const KernelViewLayout *, std::uint64_t,
    PreparedKernelRouteReservation &) noexcept;

struct BackendShape final {
  std::uint64_t storage_alignment{1u};
  std::uint64_t max_dispatch_groups{};
  std::uint64_t reset_dispatch_window{};
  std::uint64_t template_capacity{};
  std::uint64_t route_header_bytes{};
  std::uint64_t route_step_bytes{};
  std::uint64_t route_inline_step_capacity{};
  std::uint64_t template_header_bytes{};
  std::uint64_t template_step_bytes{};
  std::uint64_t template_step_capacity{
      std::numeric_limits<std::uint64_t>::max()};
  StepStructurePlan plan_step{};
};

} // namespace rund::node::accel::detail::backend_template_plan
