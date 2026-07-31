#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>

#include <cstddef>
#include <vector>

namespace node_accel_contract::runtime_case {

[[nodiscard]] inline std::vector<rund::kernel::ComputeDispatchWindow>
DispatchWindows(const rund::kernel::ComputePlan& plan) {
  std::vector<rund::kernel::ComputeDispatchWindow> windows{};
  windows.reserve(static_cast<std::size_t>(plan.dispatch_count));
  rund::kernel::u64 begin = 0u;
  while (begin < plan.tile_count) {
    const rund::kernel::u64 remaining = plan.tile_count - begin;
    const rund::kernel::u64 count =
        remaining < plan.dispatch_window_tiles ? remaining
                                               : plan.dispatch_window_tiles;
    windows.push_back(rund::kernel::ComputeDispatchWindow{
        .begin_sequence = begin,
        .tile_count = count,
    });
    begin += count;
  }
  return windows;
}

}  // namespace node_accel_contract::runtime_case
