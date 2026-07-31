#pragma once

#include <kernel/program/model.hpp>
#include <kernel/schedule/workspace.hpp>

#include <chrono>
#include <span>

namespace rund::kernel::program_detail {

using TimePoint = std::chrono::steady_clock::time_point;

struct CurrentKernelProgramViews {
  ScheduleView schedule{};
  FoldGraphView fold_graph{};
  const u32* ordered_packet_indices = nullptr;
  u32 ordered_packet_count = 0u;
  std::span<const u64> resolved_work_units{};
  KernelProgramPlacementMetadata placement{};
};

} // namespace rund::kernel::program_detail
