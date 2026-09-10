#pragma once

#include <node/accel/cpu/simd.hpp>

#include <kernel/program/compute/lowering/parse.hpp>

#include "../../local.hpp"

#include <src/accel/cpu/simd/dispatch.hpp>

#include <math32/simd/model.hpp>
#include <math64/simd/model.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace node_accel_contract::vector_plan {

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

} // namespace node_accel_contract::vector_plan
