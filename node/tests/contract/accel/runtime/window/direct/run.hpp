#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <node/accel/buffer.hpp>

#include "reject.hpp"

#include <iostream>

namespace node_accel_contract::runtime::window::direct {

[[nodiscard]] inline bool RunsAcceptedOneSubmit(const rund::AccelDevice &pick,
                                                Work &work,
                                                const Buffers &buffers) {
  const std::array<rund::kernel::ComputeDispatchWindow, 1u> full{
      rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                          .tile_count = work.plan.tile_count},
  };
  rund::node::accel::ResetRuntimeStats(pick);
  const bool accepted = pick.backend.execute(
      pick.backend.context, work.plan, work.artifact, full.data(),
      static_cast<rund::kernel::u64>(full.size()), buffers.bindings);
  if (!accepted || std::string_view{BackendLastError(pick)} != "ok") {
    std::cerr << "resident window direct execute rejected: "
              << BackendLastError(pick) << '\n';
    return false;
  }

  const auto stats = rund::node::accel::ReadRuntimeStats(pick);
  const bool metal_used_one_output_buffer =
      pick.api != rund::AccelApi::Metal ||
      stats.buffer_allocation_count + stats.buffer_reuse_hit_count == 1u;
  const auto download = rund::node::accel::DownloadBuffer(
      pick, buffers.output, work.out.data(), sizeof(work.out));
  const bool matches = stats.ok && stats.dispatch_count == 1u &&
                       metal_used_one_output_buffer && download.ok &&
                       work.out == work.expected;
  if (!matches) {
    std::cerr << "resident window mismatch: stats_ok=" << stats.ok
              << " dispatch=" << stats.dispatch_count
              << " allocations=" << stats.buffer_allocation_count
              << " reuse=" << stats.buffer_reuse_hit_count
              << " download=" << download.reason << '\n';
  }
  return matches;
}

[[nodiscard]] inline bool
DirectBackendAcceptsResidentIdentitySequence(const rund::AccelDevice &pick) {
  if (!pick.check.ok) {
    return PickUnavailableReasonIsPrecise(pick, pick.api);
  }
  if (pick.caps.max_window_tiles < 4u) {
    std::cerr << "resident window capacity too small\n";
    return false;
  }

  Work work{};
  FillInputs(work);
  const rund::compute_dsl::ComputeOp op = BuildOp(work);
  Buffers buffers{};
  if (!PrepareWork(pick, op, work)) {
    std::cerr << "resident window work preparation failed: plan="
              << work.plan.reason << " artifact=" << work.artifact.reason
              << " api=" << static_cast<unsigned>(work.plan.api) << '\n';
    return false;
  }
  if (!PrepareBuffers(pick, op, work, buffers)) {
    std::cerr << "resident window buffer preparation failed: lhs="
              << buffers.lhs.check.reason << " rhs=" << buffers.rhs.check.reason
              << " output=" << buffers.output.check.reason
              << " bindings=" << buffers.bindings.reason << '\n';
    return false;
  }

  std::vector<rund::kernel::ComputeDispatchWindow> windows =
      DispatchWindows(work.plan);
  if (windows.size() != 3u) {
    std::cerr << "resident window count mismatch: " << windows.size() << '\n';
    return false;
  }
  if (!RejectsNoncanonicalWindows(pick, work, buffers)) {
    std::cerr << "resident window accepted noncanonical sequence\n";
    return false;
  }
  if (!RejectsForgedFullRangePlan(pick, work, buffers)) {
    std::cerr << "resident window accepted forged full-range plan\n";
    return false;
  }
  if (!RejectsForgedWindow(pick, work, buffers, windows)) {
    std::cerr << "resident window accepted forged window\n";
    return false;
  }
  if (!RejectsInvalidBindings(pick, work, buffers, windows)) {
    std::cerr << "resident window accepted invalid bindings\n";
    return false;
  }
  return RunsAcceptedOneSubmit(pick, work, buffers);
}

} // namespace node_accel_contract::runtime::window::direct

namespace node_accel_contract {

[[nodiscard]] bool
ResidentWindowCollapseContract(const rund::AccelDevice &pick) {
  return runtime::window::direct::DirectBackendAcceptsResidentIdentitySequence(
      pick);
}

} // namespace node_accel_contract
