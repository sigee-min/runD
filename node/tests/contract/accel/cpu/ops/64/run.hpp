#pragma once

#include <node/accel/cpu/simd.hpp>

#include "op.hpp"
#include "ref.hpp"

#include <cstdio>

namespace node_accel_contract {

[[nodiscard]] bool RunsFixedLane64AllOps() {
  cpu::ops64::Work work = cpu::ops64::MakeWork();
  const rund::compute_dsl::ComputeOp op = cpu::ops64::BuildOp(work);

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  const rund::kernel::LoweringArtifact artifact =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Cpu);
  const rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i64>(
      25u, caps.fixed_lane64_lanes, rund::kernel::ComputeApi::Metal);
  const rund::node::accel::CpuSimdRunResult run =
      rund::node::accel::RunCpuSimd(op.ir(), caps, artifact, bindings);

  if (!run.ok) {
    std::fprintf(stderr, "ops64 run rejected reason=%s\n", run.reason);
    return false;
  }
  for (std::size_t index = 0u; index < cpu::ops64::kTileCount; ++index) {
    const auto expected = cpu::ops64::ExpectedLane(work, index);
    if (work.out[index] != expected) {
      std::fprintf(stderr, "ops64 index=%zu actual=%lld expected=%lld\n", index,
                   static_cast<long long>(work.out[index]),
                   static_cast<long long>(expected));
      std::fprintf(
          stderr,
          "  groups=scalar,bit,arithmetic,nonlinear "
          "values=%lld,%lld,%lld,%lld\n",
          static_cast<long long>(cpu::ops64::ScalarResult(work, index)),
          static_cast<long long>(cpu::ops64::BitResult(work, index)),
          static_cast<long long>(cpu::ops64::ArithmeticResult(work, index)),
          static_cast<long long>(cpu::ops64::NonlinearResult(work, index)));
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
