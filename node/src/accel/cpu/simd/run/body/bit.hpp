#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] Vec ShiftArithmeticRight(const Vec value,
                                       const u32 amount) noexcept {
  if (amount == 0u) {
    return value;
  }
  const BitsVec shifted = Bits(value) >> amount;
  const BitsVec fill = RUND_CPU_SIMD_SELECT(
      RUND_CPU_SIMD_LT(value, RUND_CPU_SIMD_SPLAT(0)),
      RUND_CPU_SIMD_SPLAT_BITS(~BitsScalar{0u}
                               << (RUND_CPU_SIMD_BITS - amount)),
      RUND_CPU_SIMD_SPLAT_BITS(0u));
  return SignedBits(shifted | fill);
}

inline void ExecuteBitAnd(const Instruction &instruction, const PreparedRun &,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  values[instruction.value_index] = SignedBits(
      Bits(values[instruction.node.lhs]) & Bits(values[instruction.node.rhs]));
}

inline void ExecuteBitOr(const Instruction &instruction, const PreparedRun &,
                         const CpuSimdBindingView &, u64, std::size_t,
                         Values &values) noexcept {
  values[instruction.value_index] = SignedBits(
      Bits(values[instruction.node.lhs]) | Bits(values[instruction.node.rhs]));
}

inline void ExecuteBitXor(const Instruction &instruction, const PreparedRun &,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  values[instruction.value_index] = SignedBits(
      Bits(values[instruction.node.lhs]) ^ Bits(values[instruction.node.rhs]));
}

inline void ExecuteBitNot(const Instruction &instruction, const PreparedRun &,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  values[instruction.value_index] =
      SignedBits(~Bits(values[instruction.node.lhs]));
}

inline void ExecuteShlConst(const Instruction &instruction, const PreparedRun &,
                            const CpuSimdBindingView &, u64, std::size_t,
                            Values &values) noexcept {
  values[instruction.value_index] =
      SignedBits(Bits(values[instruction.node.lhs]) << instruction.node.aux);
}

inline void ExecuteShrLogicalConst(const Instruction &instruction,
                                   const PreparedRun &,
                                   const CpuSimdBindingView &, u64, std::size_t,
                                   Values &values) noexcept {
  values[instruction.value_index] =
      SignedBits(Bits(values[instruction.node.lhs]) >> instruction.node.aux);
}

inline void ExecuteShrArithmeticConst(const Instruction &instruction,
                                      const PreparedRun &,
                                      const CpuSimdBindingView &, u64,
                                      std::size_t, Values &values) noexcept {
  values[instruction.value_index] =
      ShiftArithmeticRight(values[instruction.node.lhs], instruction.node.aux);
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
