#pragma once

#include <node/accel/cpu/simd.hpp>

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

template <class ValueVec, std::size_t LaneCount>
[[nodiscard]] constexpr std::size_t
IndependentScratchBytes(const std::size_t instruction_count) noexcept {
  const std::size_t values = instruction_count + 1u;
  return values * sizeof(std::uint8_t) + values * sizeof(ValueVec) +
         values * LaneCount * sizeof(__int128_t) + alignof(ValueVec) +
         alignof(__int128_t) + alignof(std::uint8_t);
}

[[nodiscard]] constexpr std::size_t
ScratchWords(const std::size_t bytes) noexcept {
  return (bytes + sizeof(std::max_align_t) - 1u) / sizeof(std::max_align_t);
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
      if (instruction.binding_slot == 0u) {
        TEST_ASSERT(instruction.full_executor_slot ==
                    kCpuSimdReadFullExecutorSlot);
        saw_contiguous_read = true;
      } else if (instruction.binding_slot == 1u) {
        TEST_ASSERT(instruction.full_executor_slot ==
                    kCpuSimdReadStridedFullExecutorSlot);
        saw_strided_read = true;
      } else {
        TEST_ASSERT(false);
      }
    } else if (instruction_op == IrOp::Write) {
      ++write_count;
      if (instruction.binding_slot == 0u) {
        TEST_ASSERT(instruction.full_executor_slot ==
                    kCpuSimdWriteFullExecutorSlot);
        saw_contiguous_write = true;
      } else if (instruction.binding_slot == 1u) {
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
          dispatch.prepared.instructions.size());
  const std::size_t lane64_scratch =
      IndependentScratchBytes<rund::math64::simd::I64x,
                              rund::math64::simd::LaneCount>(
          dispatch.prepared.instructions.size());
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
  return true;
}

} // namespace node_accel_contract
