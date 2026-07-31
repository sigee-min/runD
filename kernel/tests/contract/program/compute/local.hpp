#pragma once

#include "contract/program/compute/lowering/support.hpp"

#include <kernel/program/compute/cpu.hpp>

namespace program_compute_contract {

int RunComputeTileContract();
int RunComputeTileAsyncContract();

[[nodiscard]] constexpr rund::kernel::CpuCaps Avx2Caps() noexcept {
  return rund::kernel::CpuCaps{
      .backend = rund::kernel::ComputeBackend::Cpu,
      .strategy = rund::kernel::CpuSimdStrategy::Avx2,
      .lane_bytes = 32u,
      .fixed_lane32_lanes = 8u,
      .fixed_lane64_lanes = 4u,
      .ok = true,
      .reason = "ok",
  };
}

} // namespace program_compute_contract
