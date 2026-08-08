#pragma once

#include <node/accel/cpu/simd.hpp>

#include <kernel/program/compute/lowering/parse.hpp>
#include <kernel/program/compute/lowering/resource.hpp>

#include "../local.hpp"

#include <src/accel/cpu/simd/dispatch.hpp>

#include <math32/simd/model.hpp>
#include <math64/simd/model.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

namespace node_accel_contract {
namespace {

using rund::node::accel::cpu_simd_detail::PreparedInstruction;

static_assert(std::is_trivially_copyable_v<PreparedInstruction>);
static_assert(sizeof(PreparedInstruction) == 56u);
static_assert(sizeof(PreparedInstruction::full_executor_slot) == 1u);
static_assert(sizeof(PreparedInstruction::tail_executor_slot) == 1u);
static_assert(rund::node::accel::cpu_simd_detail::kCpuSimdBaseExecutorCount ==
              static_cast<std::size_t>(rund::kernel::IrOp::ReadUniform) + 1u);
static_assert(rund::node::accel::cpu_simd_detail::kCpuSimdExecutorCount ==
              rund::node::accel::cpu_simd_detail::kCpuSimdBaseExecutorCount +
                  3u);

[[nodiscard]] constexpr bool PackedInstructionFieldsRoundTrip() noexcept {
  PreparedInstruction instruction{};
  instruction.set_operand_fraction(0u, 7u);
  instruction.set_operand_fraction(1u, 19u);
  instruction.set_operand_fraction(2u, 31u);
  instruction.set_binding_slot(0xfedcba98u);
  return instruction.binding_slot() == 0xfedcba98u &&
         instruction.operand_fraction(0u) == 7u &&
         instruction.operand_fraction(1u) == 19u &&
         instruction.operand_fraction(2u) == 31u &&
         instruction.operand_fraction(3u) == 0u;
}

static_assert(PackedInstructionFieldsRoundTrip());

template <class ValueVec, std::size_t LaneCount>
[[nodiscard]] constexpr std::size_t
IndependentScratchBytes(const std::size_t slot_count) noexcept {
  return slot_count * sizeof(std::uint8_t) + slot_count * sizeof(ValueVec) +
         slot_count * LaneCount * sizeof(__int128_t) + alignof(ValueVec) - 1u;
}

template <class ValueVec>
[[nodiscard]] constexpr std::size_t
IndependentIntegerScratchBytes(const std::size_t slot_count) noexcept {
  return slot_count * sizeof(ValueVec) + alignof(ValueVec) - 1u;
}

[[nodiscard]] constexpr std::size_t
ScratchWords(const std::size_t bytes) noexcept {
  return (bytes + sizeof(std::max_align_t) - 1u) / sizeof(std::max_align_t);
}

struct IndependentSlotLowerBound final {
  std::size_t peak = 0u;
  std::size_t once_count = 0u;
  bool ok = false;
};

[[nodiscard]] IndependentSlotLowerBound
IndependentPhysicalSlotLowerBound(const rund::kernel::ComputeIR &ir) {
  using rund::kernel::IrOp;
  using rund::kernel::compute_lowering_detail::ParsedNodeResourcesFor;

  const auto parsed = rund::kernel::compute_lowering_detail::ParseComputeIR(ir);
  if (!parsed.ok) {
    return {};
  }
  const std::size_t count = parsed.nodes.size();
  std::vector<bool> stable(count + 1u, false);
  std::size_t once_count = 0u;
  for (std::size_t index = 0u; index < count; ++index) {
    const auto &node = parsed.nodes[index];
    const auto op = static_cast<IrOp>(node.op);
    const auto resources = ParsedNodeResourcesFor(node);
    if (!resources.ok) {
      return {};
    }
    bool is_stable =
        op == IrOp::Param || op == IrOp::Constant || op == IrOp::ReadUniform;
    if (op != IrOp::Read && op != IrOp::ReadAt && op != IrOp::Write &&
        op != IrOp::Index && !is_stable) {
      is_stable = resources.produces_value;
      for (rund::kernel::u32 ref = 0u; ref < resources.ref_count; ++ref) {
        const auto value = resources.refs[ref];
        is_stable = is_stable && value < stable.size() &&
                    (value == 0u || stable[static_cast<std::size_t>(value)]);
      }
    }
    stable[index + 1u] = is_stable;
    once_count += is_stable ? 1u : 0u;
  }

  std::vector<std::size_t> order;
  order.reserve(count);
  for (const bool stable_group : {true, false}) {
    for (std::size_t index = 0u; index < count; ++index) {
      if (stable[index + 1u] == stable_group) {
        order.push_back(index);
      }
    }
  }
  std::vector<std::size_t> last_use(count + 1u, 0u);
  std::vector<bool> pinned(count + 1u, false);
  for (std::size_t position = 0u; position < order.size(); ++position) {
    const std::size_t node_index = order[position];
    last_use[node_index + 1u] = position;
    const auto resources = ParsedNodeResourcesFor(parsed.nodes[node_index]);
    for (rund::kernel::u32 ref = 0u; ref < resources.ref_count; ++ref) {
      const auto value = resources.refs[ref];
      if (value == 0u || value >= last_use.size()) {
        return {};
      }
      last_use[value] = std::max(last_use[value], position);
      pinned[value] =
          pinned[value] || (position >= once_count && stable[value]);
    }
  }
  for (std::size_t value = 1u; value < pinned.size(); ++value) {
    if (pinned[value]) {
      last_use[value] = order.size();
    }
  }

  std::vector<bool> live(count + 1u, false);
  std::size_t live_count = 0u;
  std::size_t peak = 0u;
  for (std::size_t position = 0u; position < order.size(); ++position) {
    const std::size_t node_index = order[position];
    const auto resources = ParsedNodeResourcesFor(parsed.nodes[node_index]);
    std::array<rund::kernel::u32, 3u> unique{};
    std::size_t unique_count = 0u;
    bool dying_operand = false;
    for (rund::kernel::u32 ref = 0u; ref < resources.ref_count; ++ref) {
      const auto value = resources.refs[ref];
      bool duplicate = false;
      for (std::size_t seen = 0u; seen < unique_count; ++seen) {
        duplicate = duplicate || unique[seen] == value;
      }
      if (!duplicate) {
        unique[unique_count++] = value;
        dying_operand =
            dying_operand || (last_use[value] == position && !pinned[value]);
      }
    }
    peak = std::max(
        peak, live_count + static_cast<std::size_t>(resources.produces_value &&
                                                    !dying_operand));
    for (std::size_t ref = 0u; ref < unique_count; ++ref) {
      const auto value = unique[ref];
      if (last_use[value] == position && !pinned[value] && live[value]) {
        live[value] = false;
        --live_count;
      }
    }
    const std::size_t value = node_index + 1u;
    if (resources.produces_value && last_use[value] > position) {
      live[value] = true;
      ++live_count;
    }
  }
  return IndependentSlotLowerBound{
      .peak = peak, .once_count = once_count, .ok = true};
}

} // namespace

[[nodiscard]] bool CpuSimdFreezesExecutorSelectorsAndScratchBoundary() {
  using namespace rund::node::accel::cpu_simd_detail;
  using rund::kernel::IrOp;

  constexpr std::size_t kTileCount = 9u;
  constexpr rund::kernel::i32 kGap = -7777;
  constexpr std::array<rund::kernel::i32, kTileCount> kLeft{
      2, -4, 8, -16, 32, -64, 128, -256, 512};
  constexpr std::array<rund::kernel::i32, kTileCount> kRight{
      1, 3, -5, 7, -9, 11, -13, 15, -17};
  std::array<rund::kernel::i32, kTileCount * 2u> right{};
  std::array<rund::kernel::i32, kTileCount> sum{};
  std::array<rund::kernel::i32, kTileCount * 2u> difference{};
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    right[index * 2u] = kRight[index];
    right[index * 2u + 1u] = kGap;
    difference[index * 2u] = kGap;
    difference[index * 2u + 1u] = kGap;
  }

  const auto body = rund::compute_dsl::bind(kTileCount)
                        .fixed<1, 31>()
                        .read<"left">(kLeft.data())
                        .read<"right">(right.data())
                        .write<"sum">(sum.data())
                        .write<"difference">(difference.data());
  const rund::compute_dsl::ComputeOp op =
      rund::compute_dsl::def("node-cpu-simd-prepared-selectors")
          .on(body)
          .map([](auto i, auto b) {
            const auto left = b.template read<"left">();
            const auto right = b.template read<"right">();
            const auto sum = b.template write<"sum">();
            const auto difference = b.template write<"difference">();
            sum[i] = left[i] + right[i];
            difference[i] = left[i] - right[i];
          });
  TEST_ASSERT(op.ok());

  const rund::kernel::CpuCaps caps = cpu::NeonCaps();
  rund::kernel::BindingSet bindings = op.bindings<rund::kernel::i32>(
      31u, caps.fixed_lane32_lanes, rund::kernel::ComputeApi::Cpu);
  std::array<rund::kernel::BufferSpan, 2u> inputs{bindings.input_buffers[0],
                                                  bindings.input_buffers[1]};
  inputs[1].stride_bytes = sizeof(rund::kernel::i32) * 2u;
  bindings.input_buffers = inputs.data();
  std::array<rund::kernel::OutputSpan, 2u> outputs{bindings.output_buffers[0],
                                                   bindings.output_buffers[1]};
  outputs[1].stride_bytes = sizeof(rund::kernel::i32) * 2u;
  bindings.output_buffers = outputs.data();

  const CpuSimdDispatch dispatch =
      PrepareCpuSimdDispatch(op.ir(), caps, bindings);
  TEST_ASSERT(dispatch.prepared.ok);
  TEST_ASSERT(dispatch.run != nullptr);
  TEST_ASSERT(dispatch.scratch_bytes != nullptr);
  TEST_ASSERT(dispatch.prepared.write_count == 2u);
  const IndependentSlotLowerBound fixed_lower_bound =
      IndependentPhysicalSlotLowerBound(op.ir());
  TEST_ASSERT(fixed_lower_bound.ok);
  TEST_ASSERT(fixed_lower_bound.peak == dispatch.prepared.value_slot_count);
  TEST_ASSERT(fixed_lower_bound.once_count == dispatch.prepared.once_count);

  std::size_t read_count = 0u;
  std::size_t write_count = 0u;
  bool saw_contiguous_read = false;
  bool saw_strided_read = false;
  bool saw_contiguous_write = false;
  bool saw_strided_write = false;
  for (const PreparedInstruction &instruction :
       dispatch.prepared.instructions) {
    const CpuSimdExecutorSlot base =
        CpuSimdBaseExecutorSlot(instruction.node.op);
    TEST_ASSERT(CpuSimdExecutorSlotValid(instruction.full_executor_slot));
    TEST_ASSERT(CpuSimdExecutorSlotValid(instruction.tail_executor_slot));
    TEST_ASSERT(instruction.tail_executor_slot == base);
    const IrOp instruction_op = static_cast<IrOp>(instruction.node.op);
    if (instruction_op == IrOp::Read) {
      ++read_count;
      if (instruction.binding_slot() == 0u) {
        TEST_ASSERT(instruction.full_executor_slot ==
                    kCpuSimdReadFullExecutorSlot);
        saw_contiguous_read = true;
      } else if (instruction.binding_slot() == 1u) {
        TEST_ASSERT(instruction.full_executor_slot ==
                    kCpuSimdReadStridedFullExecutorSlot);
        saw_strided_read = true;
      } else {
        TEST_ASSERT(false);
      }
    } else if (instruction_op == IrOp::Write) {
      ++write_count;
      if (instruction.binding_slot() == 0u) {
        TEST_ASSERT(instruction.full_executor_slot ==
                    kCpuSimdWriteFullExecutorSlot);
        saw_contiguous_write = true;
      } else if (instruction.binding_slot() == 1u) {
        TEST_ASSERT(instruction.full_executor_slot == base);
        saw_strided_write = true;
      } else {
        TEST_ASSERT(false);
      }
    } else {
      TEST_ASSERT(instruction.full_executor_slot == base);
    }
  }
  TEST_ASSERT(read_count == 2u);
  TEST_ASSERT(write_count == 2u);
  TEST_ASSERT(saw_contiguous_read && saw_strided_read);
  TEST_ASSERT(saw_contiguous_write && saw_strided_write);

  const std::size_t raw_scratch =
      IndependentScratchBytes<rund::math32::simd::I32x,
                              rund::math32::simd::LaneCount>(
          dispatch.prepared.value_slot_count);
  const std::size_t lane64_scratch =
      IndependentScratchBytes<rund::math64::simd::I64x,
                              rund::math64::simd::LaneCount>(
          dispatch.prepared.value_slot_count);
  // The two reads and the first arithmetic result are the exact peak. The
  // result then destructively reuses a dying operand slot; explicit
  // quantization and the second result reuse their dying input across format
  // boundaries. Writes own no value slot.
  TEST_ASSERT(dispatch.prepared.value_slot_count == 3u);
  TEST_ASSERT(dispatch.prepared.value_slot_count <
              dispatch.prepared.instructions.size() + 1u);
  TEST_ASSERT(dispatch.scratch_bytes(dispatch.prepared) == raw_scratch);
  TEST_ASSERT(ScratchBytesFixedLane64(dispatch.prepared) == lane64_scratch);

  const std::size_t scratch_words = ScratchWords(raw_scratch);
  TEST_ASSERT(scratch_words > 1u);
  std::vector<std::max_align_t> scratch(scratch_words);
  CpuSimdBindingStorage binding_storage{};
  const CpuSimdBindingView binding_view =
      BindingView(bindings, binding_storage);
  const CpuSimdInvocation invocation{
      .bindings = &binding_view,
      .count = bindings.tile_count,
  };
  const rund::node::accel::CpuSimdRunResult exact =
      dispatch.run(dispatch.prepared, invocation,
                   CpuSimdScratch{scratch.data(),
                                  scratch.size() * sizeof(std::max_align_t)});
  TEST_ASSERT(exact.ok);
  TEST_ASSERT(exact.vector_chunk_count == 2u);
  TEST_ASSERT(exact.tail_chunk_count == 1u);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    TEST_ASSERT(sum[index] == kLeft[index] + kRight[index]);
    TEST_ASSERT(difference[index * 2u] == kLeft[index] - kRight[index]);
    TEST_ASSERT(right[index * 2u + 1u] == kGap);
    TEST_ASSERT(difference[index * 2u + 1u] == kGap);
  }

  std::fill(sum.begin(), sum.end(), kGap);
  std::fill(difference.begin(), difference.end(), kGap);
  const rund::node::accel::CpuSimdRunResult short_run = dispatch.run(
      dispatch.prepared, invocation,
      CpuSimdScratch{scratch.data(),
                     (scratch_words - 1u) * sizeof(std::max_align_t)});
  TEST_ASSERT(!short_run.ok);
  TEST_ASSERT(std::string_view{short_run.reason} == "cpu_simd_scratch_invalid");
  TEST_ASSERT(std::all_of(sum.begin(), sum.end(),
                          [](const auto value) { return value == kGap; }));
  TEST_ASSERT(std::all_of(difference.begin(), difference.end(),
                          [](const auto value) { return value == kGap; }));

  // Non-Fixed execution never consumes the 128-bit materialization plane.
  // One read followed by x+x can destructively reuse its single dying slot;
  // the duplicate operand must not release that slot twice.
  std::array<rund::kernel::i32, kTileCount> integer_input{1,  -2, 3,  -4, 5,
                                                          -6, 7,  -8, 9};
  std::array<rund::kernel::i32, kTileCount> integer_output{};
  const auto integer_body = rund::compute_dsl::bind(kTileCount)
                                .i32()
                                .read<"input">(integer_input.data())
                                .write<"output">(integer_output.data());
  const rund::compute_dsl::ComputeOp integer_op =
      rund::compute_dsl::def("node-cpu-simd-integer-scratch")
          .on(integer_body)
          .map([](auto i, auto b) {
            const auto input = b.template read<"input">();
            const auto output = b.template write<"output">();
            const auto value = input[i];
            output[i] = value + value;
          });
  TEST_ASSERT(integer_op.ok());
  const rund::kernel::BindingSet integer_bindings =
      integer_op.bindings<rund::kernel::i32>(0u, caps.fixed_lane32_lanes,
                                             rund::kernel::ComputeApi::Cpu);
  const CpuSimdDispatch integer_dispatch =
      PrepareCpuSimdDispatch(integer_op.ir(), caps, integer_bindings);
  TEST_ASSERT(integer_dispatch.prepared.ok);
  const IndependentSlotLowerBound integer_lower_bound =
      IndependentPhysicalSlotLowerBound(integer_op.ir());
  TEST_ASSERT(integer_lower_bound.ok);
  TEST_ASSERT(integer_lower_bound.peak ==
              integer_dispatch.prepared.value_slot_count);
  TEST_ASSERT(integer_dispatch.prepared.value_slot_count == 1u);
  const std::size_t integer_raw_scratch =
      IndependentIntegerScratchBytes<rund::math32::simd::I32x>(1u);
  TEST_ASSERT(integer_dispatch.scratch_bytes(integer_dispatch.prepared) ==
              integer_raw_scratch);
  const std::size_t integer_scratch_words = ScratchWords(integer_raw_scratch);
  std::vector<std::max_align_t> integer_scratch(integer_scratch_words);
  CpuSimdBindingStorage integer_binding_storage{};
  const CpuSimdBindingView integer_binding_view =
      BindingView(integer_bindings, integer_binding_storage);
  const CpuSimdInvocation integer_invocation{
      .bindings = &integer_binding_view,
      .count = integer_bindings.tile_count,
  };
  const rund::node::accel::CpuSimdRunResult integer_run = integer_dispatch.run(
      integer_dispatch.prepared, integer_invocation,
      CpuSimdScratch{
          integer_scratch.data(),
          integer_scratch.size() * sizeof(std::max_align_t),
      });
  TEST_ASSERT(integer_run.ok);
  for (std::size_t index = 0u; index < kTileCount; ++index) {
    TEST_ASSERT(integer_output[index] == integer_input[index] * 2);
  }
  std::fill(integer_output.begin(), integer_output.end(), kGap);
  const rund::node::accel::CpuSimdRunResult integer_short =
      integer_dispatch.run(
          integer_dispatch.prepared, integer_invocation,
          CpuSimdScratch{
              integer_scratch.data(),
              (integer_scratch_words - 1u) * sizeof(std::max_align_t),
          });
  TEST_ASSERT(!integer_short.ok);
  TEST_ASSERT(std::string_view{integer_short.reason} ==
              "cpu_simd_scratch_invalid");
  TEST_ASSERT(std::all_of(integer_output.begin(), integer_output.end(),
                          [](const auto value) { return value == kGap; }));

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
