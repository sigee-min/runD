#pragma once

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <kernel/program/compute/cpu.hpp>
#include <kernel/program/compute/ir.hpp>

namespace rund::node::accel {

struct CpuSimdRunResult {
  bool ok = false;
  const char *reason = "cpu_simd_run_invalid";
  rund::kernel::CpuSimdStrategy strategy =
      rund::kernel::CpuSimdStrategy::Scalar;
  rund::kernel::u64 processed_tiles = 0u;
  rund::kernel::u64 vector_chunk_count = 0u;
  rund::kernel::u64 tail_chunk_count = 0u;
  rund::kernel::u64 rejected_count = 0u;

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

[[nodiscard]] CpuSimdRunResult
RunCpuSimd(const rund::kernel::ComputeIR &ir, const rund::kernel::CpuCaps &caps,
           const rund::kernel::LoweringArtifact &artifact,
           const rund::kernel::BindingSet &bindings);

} // namespace rund::node::accel
