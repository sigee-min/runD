#pragma once

#include "context.hpp"
#include "dispatch/scratch.hpp"

#include <array>

namespace rund::kernel::compute_lowering_detail {
struct ComputeInputAdmission;
} // namespace rund::kernel::compute_lowering_detail

namespace rund::node::accel::cpu_simd_detail {

using CpuSimdRunFn = CpuSimdRunResult (*)(const PreparedRun &prepared,
                                          const CpuSimdInvocation &invocation,
                                          CpuSimdScratch scratch);
struct CpuSimdDispatch {
  PreparedRun prepared{};
  CpuSimdRunFn run = nullptr;
  CpuSimdScratchBytesFn scratch_bytes = nullptr;
};

[[nodiscard]] inline CpuSimdRunFn
SelectCpuSimdRunner(const rund::kernel::ComputeScalar scalar) noexcept {
  using rund::kernel::ComputeScalar;
  static constexpr std::array<CpuSimdRunFn, 3u> kRunnerByScalar{
      nullptr, RunFixedLane32, RunFixedLane64};
  const auto slot = static_cast<std::size_t>(scalar);
  return slot < kRunnerByScalar.size() ? kRunnerByScalar[slot] : nullptr;
}

[[nodiscard]] CpuSimdDispatch
PrepareCpuSimdDispatch(const rund::kernel::ComputeIR &ir,
                       const rund::kernel::CpuCaps &caps,
                       const rund::kernel::BindingSet &bindings);

[[nodiscard]] CpuSimdDispatch PrepareCpuSimdDispatch(
    const rund::kernel::ComputeIR &ir, const rund::kernel::CpuCaps &caps,
    const rund::kernel::compute_lowering_detail::ComputeInputAdmission &input,
    const rund::kernel::BindingSet &bindings);

[[nodiscard]] CpuSimdDispatch
PrepareCpuSimdDispatch(const rund::kernel::ComputeIR &ir,
                       const rund::kernel::CpuCaps &caps,
                       const rund::kernel::LoweringArtifact &artifact,
                       const rund::kernel::BindingSet &bindings);

} // namespace rund::node::accel::cpu_simd_detail
