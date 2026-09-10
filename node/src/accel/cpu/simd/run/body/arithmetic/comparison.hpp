#pragma once

inline void ExecuteMin(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values.set_raw(instruction.value_index,
                   ValueMin(prepared, values[instruction.node.lhs],
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
    result[lane] = lhs < rhs ? lhs : rhs;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMax(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values.set_raw(instruction.value_index,
                   ValueMax(prepared, values[instruction.node.lhs],
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
    result[lane] = lhs > rhs ? lhs : rhs;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMinUnsigned(const Instruction &instruction,
                               const PreparedRun &, const CpuSimdBindingView &,
                               u64, std::size_t, Values &values) noexcept {
  const Vec lhs = values[instruction.node.lhs];
  const Vec rhs = values[instruction.node.rhs];
  values.set_raw(instruction.value_index,
                 RUND_CPU_SIMD_VALUE_SELECT(RUND_CPU_SIMD_LT_UNSIGNED(lhs, rhs),
                                            lhs, rhs));
}

inline void ExecuteMaxUnsigned(const Instruction &instruction,
                               const PreparedRun &, const CpuSimdBindingView &,
                               u64, std::size_t, Values &values) noexcept {
  const Vec lhs = values[instruction.node.lhs];
  const Vec rhs = values[instruction.node.rhs];
  values.set_raw(instruction.value_index,
                 RUND_CPU_SIMD_VALUE_SELECT(RUND_CPU_SIMD_GT_UNSIGNED(lhs, rhs),
                                            lhs, rhs));
}

inline void ExecuteClamp(const Instruction &instruction,
                         const PreparedRun &prepared,
                         const CpuSimdBindingView &, u64, std::size_t,
                         Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    const Vec lower = ValueMax(prepared, values[instruction.node.lhs],
                               values[instruction.node.rhs]);
    values.set_raw(instruction.value_index,
                   ValueMin(prepared, lower, values[instruction.node.aux]));
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const unsigned lhs_fraction =
      ValueFractionBits(instruction, instruction.node.lhs);
  const unsigned rhs_fraction =
      ValueFractionBits(instruction, instruction.node.rhs);
  const unsigned aux_fraction =
      ValueFractionBits(instruction, instruction.node.aux);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto value = AlignWideFraction(
        values.wide(instruction.node.lhs, lane), lhs_fraction, fraction);
    const auto low = AlignWideFraction(values.wide(instruction.node.rhs, lane),
                                       rhs_fraction, fraction);
    const auto high = AlignWideFraction(values.wide(instruction.node.aux, lane),
                                        aux_fraction, fraction);
    result[lane] = value < low ? low : value > high ? high : value;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteClampUnsigned(const Instruction &instruction,
                                 const PreparedRun &,
                                 const CpuSimdBindingView &, u64, std::size_t,
                                 Values &values) noexcept {
  const Vec value = values[instruction.node.lhs];
  const Vec low = values[instruction.node.rhs];
  const Vec high = values[instruction.node.aux];
  const Vec lower = RUND_CPU_SIMD_VALUE_SELECT(
      RUND_CPU_SIMD_GT_UNSIGNED(value, low), value, low);
  values.set_raw(instruction.value_index,
                 RUND_CPU_SIMD_VALUE_SELECT(
                     RUND_CPU_SIMD_LT_UNSIGNED(lower, high), lower, high));
}

inline void ExecuteSelect(const Instruction &instruction,
                          const PreparedRun &prepared,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values.set_raw(
        instruction.value_index,
        RUND_CPU_SIMD_VALUE_SELECT(Truthy(values[instruction.node.lhs]),
                                   values[instruction.node.rhs],
                                   values[instruction.node.aux]));
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const unsigned true_fraction =
      ValueFractionBits(instruction, instruction.node.rhs);
  const unsigned false_fraction =
      ValueFractionBits(instruction, instruction.node.aux);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto when_true = AlignWideFraction(
        values.wide(instruction.node.rhs, lane), true_fraction, fraction);
    const auto when_false = AlignWideFraction(
        values.wide(instruction.node.aux, lane), false_fraction, fraction);
    result[lane] =
        values.wide(instruction.node.lhs, lane) != 0 ? when_true : when_false;
  }
  values.set_wide(instruction.value_index, result);
}
