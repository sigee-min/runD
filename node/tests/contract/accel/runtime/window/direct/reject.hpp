#pragma once

#include <accel/device.hpp>

#include "buffer.hpp"

namespace node_accel_contract::runtime::window::direct {

[[nodiscard]] inline bool
RejectsNoncanonicalWindows(const rund::AccelDevice &pick, const Work &work,
                           const Buffers &buffers) {
  const std::array<rund::kernel::ComputeDispatchWindow, 3u> windows{
      rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                          .tile_count = 2u},
      rund::kernel::ComputeDispatchWindow{.begin_sequence = 2u,
                                          .tile_count = 4u},
      rund::kernel::ComputeDispatchWindow{.begin_sequence = 6u,
                                          .tile_count = 4u},
  };
  const bool accepted = pick.backend.execute(
      pick.backend.context, work.plan, work.artifact, windows.data(),
      static_cast<rund::kernel::u64>(windows.size()), buffers.bindings);
  return !accepted && std::string_view{BackendLastError(pick)} ==
                          "compute_dispatch_count_mismatch";
}

[[nodiscard]] inline bool
RejectsForgedFullRangePlan(const rund::AccelDevice &pick, const Work &work,
                           const Buffers &buffers) {
  rund::kernel::ComputePlan forged = work.plan;
  forged.dispatch_count = 1u;
  const std::array<rund::kernel::ComputeDispatchWindow, 1u> full{
      rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                          .tile_count = work.plan.tile_count},
  };
  const bool accepted = pick.backend.execute(
      pick.backend.context, forged, work.artifact, full.data(),
      static_cast<rund::kernel::u64>(full.size()), buffers.bindings);
  return !accepted &&
         std::string_view{BackendLastError(pick)} == "compute_plan_invalid";
}

[[nodiscard]] inline bool
RejectsForgedWindow(const rund::AccelDevice &pick, const Work &work,
                    const Buffers &buffers,
                    std::vector<rund::kernel::ComputeDispatchWindow> windows) {
  windows[1].begin_sequence = 0u;
  const bool accepted = pick.backend.execute(
      pick.backend.context, work.plan, work.artifact, windows.data(),
      static_cast<rund::kernel::u64>(windows.size()), buffers.bindings);
  return !accepted && std::string_view{BackendLastError(pick)} ==
                          "compute_dispatch_count_mismatch";
}

[[nodiscard]] inline bool RejectsInvalidBindings(
    const rund::AccelDevice &pick, const Work &work, const Buffers &buffers,
    const std::vector<rund::kernel::ComputeDispatchWindow> &windows) {
  rund::kernel::BindingSet invalid = buffers.bindings;
  invalid.ok = false;
  invalid.reason = "ok";
  const bool accepted = pick.backend.execute(
      pick.backend.context, work.plan, work.artifact, windows.data(),
      static_cast<rund::kernel::u64>(windows.size()), invalid);
  return !accepted && std::string_view{BackendLastError(pick)} != "ok";
}

} // namespace node_accel_contract::runtime::window::direct
