#pragma once

#include <node/accel/cpu/simd.hpp>

#include "forged.hpp"
#include "ref.hpp"

#include "../../../../../src/accel/cpu/simd/dispatch.hpp"

#include <kernel/program/compute/lowering/emission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace node_accel_contract {

[[nodiscard]] inline bool
CpuInternalPreparationIsReused(const rund::compute_dsl::ComputeOp &op,
                               const cpu::pick::Resources &resources) {
  using namespace rund::node::accel::cpu_simd_detail;
  const auto admission =
      rund::kernel::compute_lowering_detail::AdmitArtifact(
          resources.plan, resources.artifact);
  TEST_ASSERT(admission.ok);
  TEST_ASSERT(admission.parse_count() == 1u);
  TEST_ASSERT(admission.emission_count == 1u);
  auto retained =
      rund::kernel::compute_lowering_detail::LowerRetainedComputeArtifact(
          op.ir(), rund::kernel::ComputeApi::Cpu);
  TEST_ASSERT(retained.artifact.ok);
  TEST_ASSERT(retained.input.ok);
  TEST_ASSERT(retained.parse_count() == 1u);
  TEST_ASSERT(retained.emission_count == 1u);
  std::string{}.swap(retained.artifact.source_text);
  const auto warm =
      rund::kernel::compute_lowering_detail::AdmitRetained(
          resources.plan, retained.artifact, &retained.input);
  TEST_ASSERT(warm.ok);
  TEST_ASSERT(warm.parse_count == 0u);
  TEST_ASSERT(warm.emission_count == 0u);
  const CpuSimdDispatch dispatch = PrepareCpuSimdDispatch(
      op.ir(), resources.pick.cpu_caps, retained.input, resources.bindings);
  TEST_ASSERT(dispatch.prepared.ok);
  TEST_ASSERT(dispatch.run != nullptr);
  TEST_ASSERT(dispatch.scratch_bytes != nullptr);

  const std::size_t bytes = dispatch.scratch_bytes(dispatch.prepared);
  std::vector<std::max_align_t> scratch(
      (bytes + sizeof(std::max_align_t) - 1u) / sizeof(std::max_align_t));
  CpuSimdBindingStorage storage{};
  const CpuSimdBindingView view = BindingView(resources.bindings, storage);
  for (std::size_t run_index = 0u; run_index < 2u; ++run_index) {
    const rund::node::accel::CpuSimdRunResult result =
        dispatch.run(dispatch.prepared,
                     CpuSimdInvocation{.bindings = &view,
                                       .count = resources.bindings.tile_count},
                     CpuSimdScratch{scratch.data(),
                                    scratch.size() * sizeof(std::max_align_t)});
    TEST_ASSERT(result.ok);
  }
  return true;
}

[[nodiscard]] bool CpuPickRunsGenericBackendMap() {
  namespace cpu_pick = node_accel_contract::cpu::pick;

  cpu_pick::Work work{};
  const rund::compute_dsl::ComputeOp op = cpu_pick::BuildOp(work);
  const cpu_pick::Resources resources = cpu_pick::BuildResources(op);

  TEST_ASSERT(resources.pick.check.ok);
  TEST_ASSERT(resources.pick.caps.ok);
  TEST_ASSERT(resources.pick.caps.api == rund::kernel::ComputeApi::Cpu);
  TEST_ASSERT(resources.pick.cpu_caps.ok);
  TEST_ASSERT(static_cast<bool>(resources.pick.backend));
  TEST_ASSERT(resources.plan.ok);
  TEST_ASSERT(resources.artifact.ok);
  const auto admission =
      rund::kernel::compute_lowering_detail::AdmitArtifact(
          resources.plan, resources.artifact);
  TEST_ASSERT(admission.ok);
  TEST_ASSERT(admission.parse_count() == 1u);
  TEST_ASSERT(admission.emission_count == 1u);
  TEST_ASSERT(resources.pick.backend.execute(
      resources.pick.backend.context, resources.plan, resources.artifact,
      &resources.window, 1u, resources.bindings));

  for (std::size_t index = 0u; index < cpu_pick::kTileCount; ++index) {
    const rund::kernel::i32 expected =
        cpu_pick::ExpectedValue(work.lhs[index], work.rhs[index]);
    if (work.out[index] != expected) {
      std::fprintf(stderr,
                   "cpu pick mismatch index=%zu lhs=%d rhs=%d actual=%d "
                   "expected=%d\n",
                   index, work.lhs[index], work.rhs[index], work.out[index],
                   expected);
      return false;
    }
  }
  TEST_ASSERT(CpuInternalPreparationIsReused(op, resources));
  TEST_ASSERT(cpu_pick::RejectsForgedArtifacts(work, resources));
  return true;
}

} // namespace node_accel_contract
