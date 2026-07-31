#pragma once

#include <kernel/program/request.hpp>

namespace rund::kernel::internal {

constexpr bool ProgramRequiresNoGrowth(const bool request_no_growth,
                                       const AllocationPolicy allocation) {
  return request_no_growth || allocation == AllocationPolicy::NoGrowth;
}

constexpr bool ProgramRequiresNoGrowth(const KernelProgramCompileRequest& request) {
  return ProgramRequiresNoGrowth(request.require_no_allocation, request.schedule.allocation);
}

constexpr bool ProgramRequiresNoGrowth(const KernelProgramCompileRequest& request,
                                       const ScheduleCompileRequest& schedule_request) {
  return ProgramRequiresNoGrowth(request.require_no_allocation, schedule_request.allocation);
}

} // namespace rund::kernel::internal
