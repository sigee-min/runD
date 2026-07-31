#pragma once

#include <kernel/program/request.hpp>

namespace kernel_contract_test::program_no_allocation {

inline rund::kernel::ScheduleCompileRequest ScheduleRequest() {
  return rund::kernel::ScheduleCompileRequest{
      .packet_count = 12u,
      .execution_width = 3u,
      .intent = rund::kernel::PartitionIntent::StaticWidth,
      .placement = rund::kernel::PlacementPolicy::Uniform,
      .allocation = rund::kernel::AllocationPolicy::NoGrowth,
  };
}

} // namespace kernel_contract_test::program_no_allocation
