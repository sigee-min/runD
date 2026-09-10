#include "affine.hpp"
#include "layout.hpp"
#include "local.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace node_accel_contract {

[[nodiscard]] bool CheckCommitDemand() {
  using namespace rund::node::accel::cpu_simd_detail;
  using namespace vector_plan;

  for (const std::size_t stride : {1u, 2u}) {
    for (const bool in_place : {false, true}) {
      TEST_ASSERT(CheckPreparedAffine<rund::kernel::i32>(stride, in_place));
      TEST_ASSERT(CheckPreparedAffine<rund::kernel::i64>(stride, in_place));
    }
  }
  // A quadratic product must retain instruction execution, not be flattened.
  std::array<rund::kernel::i32, 4u> nonlinear_input{1, 2, 3, 4},
      nonlinear_output{};
  const auto nonlinear = rund::compute_dsl::def("cpu-non-affine-product")
                             .on(rund::compute_dsl::bind(4u)
                                     .i32()
                                     .read<"in">(nonlinear_input.data())
                                     .write<"out">(nonlinear_output.data()))
                             .map([](auto i, auto b) {
                               const auto x = b.template read<"in">()[i];
                               b.template write<"out">()[i] = x * x;
                             });
  const auto nonlinear_caps = cpu::NeonCaps();
  const auto nonlinear_bindings = nonlinear.bindings<rund::kernel::i32>(
      0u, nonlinear_caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Cpu);
  const auto nonlinear_dispatch = PrepareCpuSimdDispatch(
      nonlinear.ir(), nonlinear_caps, nonlinear_bindings);
  TEST_ASSERT(nonlinear_dispatch.prepared.ok &&
              !nonlinear_dispatch.prepared.affine.valid);
  constexpr std::size_t kTileCount = 9u;
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();

  // The lower bound is the commit-demand peak of the admitted, non-DCE
  // schedule. A pinned parameter occupies one slot while a deliberately dead
  // pure definition still needs one destination commit slot.
  std::array<rund::kernel::i32, kTileCount> commit_output{};
  const auto commit_body = rund::compute_dsl::bind(kTileCount)
                               .i32()
                               .param<"pinned">(11)
                               .write<"output">(commit_output.data());
  const rund::compute_dsl::ComputeOp commit_op =
      rund::compute_dsl::def("node-cpu-simd-commit-demand")
          .on(commit_body)
          .map([](auto i, auto b) {
            const auto pinned = b.template param<"pinned">();
            static_cast<void>(pinned + 1);
            b.template write<"output">()[i] = pinned;
          });
  TEST_ASSERT(commit_op.ok());
  const rund::kernel::BindingSet commit_bindings =
      commit_op.bindings<rund::kernel::i32>(0u, caps.fixed_lane32_lanes,
                                            rund::kernel::ComputeApi::Cpu);
  const CpuSimdDispatch commit_dispatch =
      PrepareCpuSimdDispatch(commit_op.ir(), caps, commit_bindings);
  const IndependentSlotLowerBound commit_lower_bound =
      IndependentPhysicalSlotLowerBound(commit_op.ir());
  TEST_ASSERT(commit_dispatch.prepared.ok);
  TEST_ASSERT(commit_lower_bound.ok);
  TEST_ASSERT(commit_lower_bound.peak == 2u);
  TEST_ASSERT(commit_dispatch.prepared.value_slot_count ==
              commit_lower_bound.peak);
  const rund::node::accel::CpuSimdRunResult commit_run =
      rund::node::accel::RunCpuSimd(
          commit_op.ir(), caps,
          rund::kernel::LowerComputeIR(commit_op.ir(),
                                       rund::kernel::ComputeApi::Cpu),
          commit_bindings);
  TEST_ASSERT(commit_run.ok);
  TEST_ASSERT(std::all_of(commit_output.begin(), commit_output.end(),
                          [](const auto value) { return value == 11; }));
  return true;
}

} // namespace node_accel_contract
