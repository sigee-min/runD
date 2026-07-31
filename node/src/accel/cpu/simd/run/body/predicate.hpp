#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

enum class WideCompare : std::uint8_t { Eq, Ne, Lt, Le, Gt, Ge };

inline void FixedCompare(const Instruction &instruction,
                         const PreparedRun &prepared, Values &values,
                         const WideCompare comparison) noexcept {
  std::array<WideScalar, kLaneCount> lanes{};
  const auto lhs_format = ValueFormat(prepared, instruction.node.lhs);
  const auto rhs_format = ValueFormat(prepared, instruction.node.rhs);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar lhs =
        AlignWideFraction(values.wide(instruction.node.lhs, lane),
                          lhs_format.fraction_bits, fraction);
    const WideScalar rhs =
        AlignWideFraction(values.wide(instruction.node.rhs, lane),
                          rhs_format.fraction_bits, fraction);
    bool result = false;
    switch (comparison) {
    case WideCompare::Eq:
      result = lhs == rhs;
      break;
    case WideCompare::Ne:
      result = lhs != rhs;
      break;
    case WideCompare::Lt:
      result = lhs < rhs;
      break;
    case WideCompare::Le:
      result = lhs <= rhs;
      break;
    case WideCompare::Gt:
      result = lhs > rhs;
      break;
    case WideCompare::Ge:
      result = lhs >= rhs;
      break;
    }
    lanes[lane] = result ? WideScalar{1} : WideScalar{0};
  }
  values.set_wide(instruction.value_index, lanes);
}

inline void ExecuteEq(const Instruction &instruction,
                      const PreparedRun &prepared, const CpuSimdBindingView &,
                      u64, std::size_t, Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    FixedCompare(instruction, prepared, values, WideCompare::Eq);
    return;
  }
  values[instruction.value_index] = BooleanValue(RUND_CPU_SIMD_EQ(
      values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteNe(const Instruction &instruction,
                      const PreparedRun &prepared, const CpuSimdBindingView &,
                      u64, std::size_t, Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    FixedCompare(instruction, prepared, values, WideCompare::Ne);
    return;
  }
  values[instruction.value_index] = BooleanValue(RUND_CPU_SIMD_NE(
      values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteLt(const Instruction &instruction,
                      const PreparedRun &prepared, const CpuSimdBindingView &,
                      u64, std::size_t, Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    FixedCompare(instruction, prepared, values, WideCompare::Lt);
    return;
  }
  values[instruction.value_index] = BooleanValue(ValueLt(
      prepared, values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteLe(const Instruction &instruction,
                      const PreparedRun &prepared, const CpuSimdBindingView &,
                      u64, std::size_t, Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    FixedCompare(instruction, prepared, values, WideCompare::Le);
    return;
  }
  values[instruction.value_index] = BooleanValue(ValueLe(
      prepared, values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteGt(const Instruction &instruction,
                      const PreparedRun &prepared, const CpuSimdBindingView &,
                      u64, std::size_t, Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    FixedCompare(instruction, prepared, values, WideCompare::Gt);
    return;
  }
  values[instruction.value_index] = BooleanValue(ValueGt(
      prepared, values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteGe(const Instruction &instruction,
                      const PreparedRun &prepared, const CpuSimdBindingView &,
                      u64, std::size_t, Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    FixedCompare(instruction, prepared, values, WideCompare::Ge);
    return;
  }
  values[instruction.value_index] = BooleanValue(ValueGe(
      prepared, values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteLtUnsigned(const Instruction &instruction,
                              const PreparedRun &, const CpuSimdBindingView &,
                              u64, std::size_t, Values &values) noexcept {
  values[instruction.value_index] = BooleanValue(RUND_CPU_SIMD_LT_UNSIGNED(
      values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteLeUnsigned(const Instruction &instruction,
                              const PreparedRun &, const CpuSimdBindingView &,
                              u64, std::size_t, Values &values) noexcept {
  values[instruction.value_index] = BooleanValue(RUND_CPU_SIMD_LE_UNSIGNED(
      values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteGtUnsigned(const Instruction &instruction,
                              const PreparedRun &, const CpuSimdBindingView &,
                              u64, std::size_t, Values &values) noexcept {
  values[instruction.value_index] = BooleanValue(RUND_CPU_SIMD_GT_UNSIGNED(
      values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecuteGeUnsigned(const Instruction &instruction,
                              const PreparedRun &, const CpuSimdBindingView &,
                              u64, std::size_t, Values &values) noexcept {
  values[instruction.value_index] = BooleanValue(RUND_CPU_SIMD_GE_UNSIGNED(
      values[instruction.node.lhs], values[instruction.node.rhs]));
}

inline void ExecutePredicateNot(const Instruction &instruction,
                                const PreparedRun &prepared,
                                const CpuSimdBindingView &, u64, std::size_t,
                                Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    std::array<WideScalar, kLaneCount> result{};
    for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
      result[lane] = values.wide(instruction.node.lhs, lane) == 0
                         ? WideScalar{1}
                         : WideScalar{0};
    }
    values.set_wide(instruction.value_index, result);
    return;
  }
  values[instruction.value_index] =
      BooleanValue(~Truthy(values[instruction.node.lhs]));
}

inline void ExecutePredicateAnd(const Instruction &instruction,
                                const PreparedRun &prepared,
                                const CpuSimdBindingView &, u64, std::size_t,
                                Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    std::array<WideScalar, kLaneCount> result{};
    for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
      result[lane] = values.wide(instruction.node.lhs, lane) != 0 &&
                             values.wide(instruction.node.rhs, lane) != 0
                         ? WideScalar{1}
                         : WideScalar{0};
    }
    values.set_wide(instruction.value_index, result);
    return;
  }
  values[instruction.value_index] =
      BooleanValue(Truthy(values[instruction.node.lhs]) &
                   Truthy(values[instruction.node.rhs]));
}

inline void ExecutePredicateOr(const Instruction &instruction,
                               const PreparedRun &prepared,
                               const CpuSimdBindingView &, u64, std::size_t,
                               Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    std::array<WideScalar, kLaneCount> result{};
    for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
      result[lane] = values.wide(instruction.node.lhs, lane) != 0 ||
                             values.wide(instruction.node.rhs, lane) != 0
                         ? WideScalar{1}
                         : WideScalar{0};
    }
    values.set_wide(instruction.value_index, result);
    return;
  }
  values[instruction.value_index] =
      BooleanValue(Truthy(values[instruction.node.lhs]) |
                   Truthy(values[instruction.node.rhs]));
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
