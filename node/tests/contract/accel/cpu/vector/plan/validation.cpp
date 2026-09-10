#include "layout.hpp"
#include "local.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

namespace node_accel_contract {

[[nodiscard]] bool CheckStablePrefixFailure() {
  using namespace rund::node::accel::cpu_simd_detail;
  using rund::kernel::IrOp;
  using vector_plan::PreparedInstruction;
  using vector_plan::ScratchWords;

  constexpr std::size_t kTileCount = 9u;
  constexpr rund::kernel::i32 kGap = -7777;
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();

  // Stable-prefix failures are terminal at the exact failing instruction.
  // Replace the following pure dependent with a valid write: an executor that
  // keeps walking after a divide-by-zero would visibly corrupt the output.
  std::array<rund::kernel::i32, kTileCount> barrier_output{};
  barrier_output.fill(kGap);
  const auto barrier_body = rund::compute_dsl::bind(kTileCount)
                                .i32()
                                .param<"numerator">(7)
                                .param<"zero">(0)
                                .write<"output">(barrier_output.data());
  const rund::compute_dsl::ComputeOp barrier_op =
      rund::compute_dsl::def("node-cpu-simd-once-failure-barrier")
          .on(barrier_body)
          .map([](auto i, auto b) {
            const auto numerator = b.template param<"numerator">();
            const auto zero = b.template param<"zero">();
            b.template write<"output">()[i] = numerator / zero + numerator;
          });
  TEST_ASSERT(barrier_op.ok());
  const rund::kernel::BindingSet barrier_bindings =
      barrier_op.bindings<rund::kernel::i32>(0u, caps.fixed_lane32_lanes,
                                             rund::kernel::ComputeApi::Cpu);
  CpuSimdDispatch barrier_dispatch =
      PrepareCpuSimdDispatch(barrier_op.ir(), caps, barrier_bindings);
  TEST_ASSERT(barrier_dispatch.prepared.ok);
  std::size_t divide_index = barrier_dispatch.prepared.instructions.size();
  std::size_t dependent_index = barrier_dispatch.prepared.instructions.size();
  std::size_t write_index = barrier_dispatch.prepared.instructions.size();
  for (std::size_t index = 0u;
       index < barrier_dispatch.prepared.instructions.size(); ++index) {
    const IrOp current = static_cast<IrOp>(
        barrier_dispatch.prepared.instructions[index].node.op);
    if (current == IrOp::DivSigned) {
      divide_index = index;
    } else if (current == IrOp::Add &&
               index < barrier_dispatch.prepared.once_count) {
      dependent_index = index;
    } else if (current == IrOp::Write) {
      write_index = index;
    }
  }
  TEST_ASSERT(divide_index < dependent_index);
  TEST_ASSERT(dependent_index < barrier_dispatch.prepared.once_count);
  TEST_ASSERT(write_index < barrier_dispatch.prepared.instructions.size());
  const PreparedInstruction division =
      barrier_dispatch.prepared.instructions[divide_index];
  const PreparedInstruction dependent =
      barrier_dispatch.prepared.instructions[dependent_index];
  const rund::kernel::u32 retained_numerator =
      dependent.node.lhs == division.value_index ? dependent.node.rhs
                                                 : dependent.node.lhs;
  TEST_ASSERT(retained_numerator != division.value_index);
  barrier_dispatch.prepared.instructions[dependent_index] =
      barrier_dispatch.prepared.instructions[write_index];
  barrier_dispatch.prepared.instructions[dependent_index].node.lhs =
      retained_numerator;
  const std::size_t barrier_scratch_bytes =
      barrier_dispatch.scratch_bytes(barrier_dispatch.prepared);
  std::vector<std::max_align_t> barrier_scratch(
      ScratchWords(barrier_scratch_bytes));
  CpuSimdBindingStorage barrier_binding_storage{};
  const CpuSimdBindingView barrier_binding_view =
      BindingView(barrier_bindings, barrier_binding_storage);
  const CpuSimdInvocation barrier_invocation{
      .bindings = &barrier_binding_view,
      .count = barrier_bindings.tile_count,
  };
  const rund::node::accel::CpuSimdRunResult barrier_run = barrier_dispatch.run(
      barrier_dispatch.prepared, barrier_invocation,
      CpuSimdScratch{barrier_scratch.data(),
                     barrier_scratch.size() * sizeof(std::max_align_t)});
  TEST_ASSERT(!barrier_run.ok);
  TEST_ASSERT(std::string_view{barrier_run.reason} ==
              "compute_integer_divide_by_zero");
  TEST_ASSERT(std::all_of(barrier_output.begin(), barrier_output.end(),
                          [](const auto value) { return value == kGap; }));
  return true;
}

[[nodiscard]] bool CheckStableEdgeAuthority() {
  using namespace rund::node::accel::cpu_simd_detail;
  using namespace vector_plan;
  using rund::kernel::IrOp;

  constexpr std::size_t kTileCount = 9u;
  const rund::kernel::CpuCaps caps = cpu::NeonCaps();

  // A dynamic read deliberately precedes a constant shift. The shift amount
  // is an immediate, never an SSA edge; the shift and both format-changing
  // quantizations therefore belong to the stable-once prefix. Their physical
  // slot is reused across formats, then remains pinned while the dynamic read
  // occupies the only other live slot.
  std::array<rund::kernel::i32, kTileCount> dynamic_input{21, 22, 23, 24, 25,
                                                          26, 27, 28, 29};
  std::array<rund::kernel::i32, kTileCount> stable_output{};
  std::array<rund::kernel::i32, kTileCount> dynamic_output{};
  const auto stable_body = rund::compute_dsl::bind(kTileCount)
                               .fixed<16, 16>()
                               .param<"constant">(3)
                               .read<"dynamic">(dynamic_input.data())
                               .write<"stable">(stable_output.data())
                               .write<"copy">(dynamic_output.data());
  const rund::compute_dsl::ComputeOp stable_op =
      rund::compute_dsl::def("node-cpu-simd-stable-edge-authority")
          .on(stable_body)
          .map([](auto i, auto b) {
            const auto constant = b.template param<"constant">();
            const auto dynamic = b.template read<"dynamic">();
            const auto stable = b.template write<"stable">();
            const auto copy = b.template write<"copy">();
            const auto observed = dynamic[i];
            const auto shifted = rund::compute_dsl::shl_const<2>(constant);
            stable[i] = rund::compute_dsl::quantize<8, 24>(shifted);
            copy[i] = observed;
          });
  TEST_ASSERT(stable_op.ok());
  const rund::kernel::BindingSet stable_bindings =
      stable_op.bindings<rund::kernel::i32>(16u, caps.fixed_lane32_lanes,
                                            rund::kernel::ComputeApi::Cpu);
  const CpuSimdDispatch stable_dispatch =
      PrepareCpuSimdDispatch(stable_op.ir(), caps, stable_bindings);
  TEST_ASSERT(stable_dispatch.prepared.ok);
  const IndependentSlotLowerBound stable_lower_bound =
      IndependentPhysicalSlotLowerBound(stable_op.ir());
  TEST_ASSERT(stable_lower_bound.ok);
  TEST_ASSERT(stable_lower_bound.peak ==
              stable_dispatch.prepared.value_slot_count);
  TEST_ASSERT(stable_lower_bound.once_count ==
              stable_dispatch.prepared.once_count);
  TEST_ASSERT(stable_dispatch.prepared.value_slot_count == 2u);
  bool stable_shift = false;
  std::size_t stable_quantize_count = 0u;
  for (std::size_t index = 0u;
       index < stable_dispatch.prepared.instructions.size(); ++index) {
    const IrOp instruction_op =
        static_cast<IrOp>(stable_dispatch.prepared.instructions[index].node.op);
    if (instruction_op == IrOp::ShlConst) {
      TEST_ASSERT(index < stable_dispatch.prepared.once_count);
      stable_shift = true;
    } else if (instruction_op == IrOp::Quantize &&
               index < stable_dispatch.prepared.once_count) {
      ++stable_quantize_count;
    }
  }
  TEST_ASSERT(stable_shift);
  TEST_ASSERT(stable_quantize_count >= 1u);
  const rund::node::accel::CpuSimdRunResult stable_run =
      rund::node::accel::RunCpuSimd(
          stable_op.ir(), caps,
          rund::kernel::LowerComputeIR(stable_op.ir(),
                                       rund::kernel::ComputeApi::Cpu),
          stable_bindings);
  TEST_ASSERT(stable_run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    // Shifting the 16.16 raw value by two yields 12; quantizing to 8.24
    // preserves the numeric value by adding eight fractional bits.
    TEST_ASSERT(stable_output[index] == 3072);
    TEST_ASSERT(dynamic_output[index] == dynamic_input[index]);
  }
  return true;
}

} // namespace node_accel_contract
