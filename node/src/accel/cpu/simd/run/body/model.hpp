#pragma once

#include <array>
#include <bit>
#include <cstring>

#include "values.hpp"

namespace rund::node::accel::cpu_simd_detail {
namespace {

using rund::kernel::IrOp;
using rund::kernel::u32;
using rund::kernel::u64;
using Scalar = RUND_CPU_SIMD_SCALAR;
using BitsScalar = RUND_CPU_SIMD_BITS_SCALAR;
using Vec = RUND_CPU_SIMD_VEC;
using Mask = RUND_CPU_SIMD_MASK;
using BitsVec = RUND_CPU_SIMD_BITS_VEC;
inline constexpr std::size_t kLaneCount = RUND_CPU_SIMD_LANES;

using Instruction = PreparedInstruction;
using ExecuteFn = void (*)(const Instruction &instruction,
                           const PreparedRun &prepared,
                           const CpuSimdBindingView &bindings, u64 base_tile,
                           std::size_t live_lanes, Values &values) noexcept;

[[nodiscard]] inline rund::kernel::u8
ValueFractionBits(const Instruction &instruction, const u32 value) noexcept {
  if (instruction.node.lhs == value) {
    return instruction.operand_fraction(0u);
  }
  if (instruction.node.rhs == value) {
    return instruction.operand_fraction(1u);
  }
  if (instruction.node.aux == value) {
    return instruction.operand_fraction(2u);
  }
  return 0u;
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
