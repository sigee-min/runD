#pragma once

inline void ExecuteAdd(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values.set_raw(instruction.value_index,
                   RUND_CPU_SIMD_ADD_WRAP(values[instruction.node.lhs],
                                          values[instruction.node.rhs]));
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const unsigned lhs_fraction =
      ValueFractionBits(instruction, instruction.node.lhs);
  const unsigned rhs_fraction =
      ValueFractionBits(instruction, instruction.node.rhs);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto lhs = AlignWideFraction(values.wide(instruction.node.lhs, lane),
                                       lhs_fraction, fraction);
    const auto rhs = AlignWideFraction(values.wide(instruction.node.rhs, lane),
                                       rhs_fraction, fraction);
    result[lane] = lhs + rhs;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteSub(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values.set_raw(instruction.value_index,
                   RUND_CPU_SIMD_SUB_WRAP(values[instruction.node.lhs],
                                          values[instruction.node.rhs]));
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const unsigned lhs_fraction =
      ValueFractionBits(instruction, instruction.node.lhs);
  const unsigned rhs_fraction =
      ValueFractionBits(instruction, instruction.node.rhs);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto lhs = AlignWideFraction(values.wide(instruction.node.lhs, lane),
                                       lhs_fraction, fraction);
    const auto rhs = AlignWideFraction(values.wide(instruction.node.rhs, lane),
                                       rhs_fraction, fraction);
    result[lane] = lhs - rhs;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMul(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values.set_raw(instruction.value_index,
                   RUND_CPU_SIMD_MUL_LOW(values[instruction.node.lhs],
                                         values[instruction.node.rhs]));
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    result[lane] = values.wide(instruction.node.lhs, lane) *
                   values.wide(instruction.node.rhs, lane);
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMulWrap(const Instruction &instruction, const PreparedRun &,
                           const CpuSimdBindingView &, u64, std::size_t,
                           Values &values) noexcept {
  values.set_raw(instruction.value_index,
                 RUND_CPU_SIMD_MUL_LOW(values[instruction.node.lhs],
                                       values[instruction.node.rhs]));
}

inline void ExecuteQuantize(const Instruction &instruction, const PreparedRun &,
                            const CpuSimdBindingView &, u64, std::size_t,
                            Values &values) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  const rund::kernel::ComputeFixedFormat source{
      .fraction_bits = ValueFractionBits(instruction, instruction.node.lhs),
  };
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    result[lane] = QuantizeWide(values.wide(instruction.node.lhs, lane), source,
                                instruction.node.fixed_format);
  }
  values.set_wide(instruction.value_index, result);
}
