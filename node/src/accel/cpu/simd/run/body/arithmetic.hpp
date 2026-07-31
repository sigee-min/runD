#pragma once

#include <limits>

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] inline WideScalar
SignedWideFromBits(const __uint128_t bits) noexcept {
  return std::bit_cast<WideScalar>(bits);
}

[[nodiscard]] inline WideScalar ShiftWideLeft(const WideScalar value,
                                              const unsigned shift) noexcept {
  return SignedWideFromBits(static_cast<__uint128_t>(value) << shift);
}

[[nodiscard]] inline WideScalar
AlignWideFraction(const WideScalar value, const unsigned source_fraction,
                  const unsigned target_fraction) noexcept {
  return source_fraction == target_fraction
             ? value
             : ShiftWideLeft(value, target_fraction - source_fraction);
}

[[nodiscard]] inline WideScalar
QuantizeWide(const WideScalar value,
             const rund::kernel::ComputeFixedFormat source,
             const rund::kernel::ComputeFixedFormat target) noexcept {
  const unsigned width =
      static_cast<unsigned>(target.integer_bits) + target.fraction_bits;
  const __uint128_t sign = static_cast<__uint128_t>(1u) << (width - 1u);
  const WideScalar low = -static_cast<WideScalar>(sign);
  const WideScalar high = static_cast<WideScalar>(sign - 1u);
  WideScalar scaled = value;
  if (target.fraction_bits > source.fraction_bits) {
    const unsigned shift =
        static_cast<unsigned>(target.fraction_bits - source.fraction_bits);
    if (target.overflow == rund::kernel::ComputeOverflow::Saturate) {
      const WideScalar factor = static_cast<WideScalar>(1u) << shift;
      const WideScalar low_before_scale = low / factor;
      const WideScalar high_before_scale = high / factor;
      if (value < low_before_scale) {
        return low;
      }
      if (value > high_before_scale) {
        return high;
      }
    }
    scaled = ShiftWideLeft(scaled, shift);
  } else if (target.fraction_bits < source.fraction_bits) {
    const unsigned shift =
        static_cast<unsigned>(source.fraction_bits - target.fraction_bits);
    const bool negative = scaled < 0;
    const __uint128_t bits = static_cast<__uint128_t>(scaled);
    const __uint128_t magnitude = negative ? (~bits + 1u) : bits;
    __uint128_t quotient = magnitude >> shift;
    const __uint128_t mask = (static_cast<__uint128_t>(1u) << shift) - 1u;
    const __uint128_t remainder = magnitude & mask;
    const __uint128_t halfway = static_cast<__uint128_t>(1u) << (shift - 1u);
    const bool nonzero = remainder != 0u;
    const bool nearest =
        remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u);
    if ((target.rounding == rund::kernel::ComputeRounding::Down && negative &&
         nonzero) ||
        (target.rounding == rund::kernel::ComputeRounding::Up && !negative &&
         nonzero) ||
        (target.rounding == rund::kernel::ComputeRounding::NearestEven &&
         nearest)) {
      ++quotient;
    }
    scaled = negative ? SignedWideFromBits(~quotient + 1u)
                      : SignedWideFromBits(quotient);
  }

  if (target.overflow == rund::kernel::ComputeOverflow::Saturate) {
    return scaled < low ? low : scaled > high ? high : scaled;
  }
  const __uint128_t mask = width == 128u
                               ? ~static_cast<__uint128_t>(0u)
                               : (static_cast<__uint128_t>(1u) << width) - 1u;
  __uint128_t wrapped = static_cast<__uint128_t>(scaled) & mask;
  if (width < 128u && (wrapped & sign) != 0u) {
    wrapped |= ~mask;
  }
  return SignedWideFromBits(wrapped);
}

inline void ExecuteAdd(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values[instruction.value_index] = RUND_CPU_SIMD_ADD_WRAP(
        values[instruction.node.lhs], values[instruction.node.rhs]);
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const auto lhs_format = ValueFormat(prepared, instruction.node.lhs);
  const auto rhs_format = ValueFormat(prepared, instruction.node.rhs);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto lhs = AlignWideFraction(values.wide(instruction.node.lhs, lane),
                                       lhs_format.fraction_bits, fraction);
    const auto rhs = AlignWideFraction(values.wide(instruction.node.rhs, lane),
                                       rhs_format.fraction_bits, fraction);
    result[lane] = lhs + rhs;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteSub(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values[instruction.value_index] = RUND_CPU_SIMD_SUB_WRAP(
        values[instruction.node.lhs], values[instruction.node.rhs]);
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const auto lhs_format = ValueFormat(prepared, instruction.node.lhs);
  const auto rhs_format = ValueFormat(prepared, instruction.node.rhs);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto lhs = AlignWideFraction(values.wide(instruction.node.lhs, lane),
                                       lhs_format.fraction_bits, fraction);
    const auto rhs = AlignWideFraction(values.wide(instruction.node.rhs, lane),
                                       rhs_format.fraction_bits, fraction);
    result[lane] = lhs - rhs;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMul(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values[instruction.value_index] = RUND_CPU_SIMD_MUL_LOW(
        values[instruction.node.lhs], values[instruction.node.rhs]);
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
  values[instruction.value_index] = RUND_CPU_SIMD_MUL_LOW(
      values[instruction.node.lhs], values[instruction.node.rhs]);
  values.invalidate(instruction.value_index);
}

inline void ExecuteQuantize(const Instruction &instruction,
                            const PreparedRun &prepared,
                            const CpuSimdBindingView &, u64, std::size_t,
                            Values &values) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  const auto source = ValueFormat(prepared, instruction.node.lhs);
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    result[lane] = QuantizeWide(values.wide(instruction.node.lhs, lane), source,
                                instruction.node.fixed_format);
  }
  values.set_wide(instruction.value_index, result);
}

#include "arithmetic/divide.hpp"

inline void ExecuteMin(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values[instruction.value_index] = ValueMin(
        prepared, values[instruction.node.lhs], values[instruction.node.rhs]);
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const auto lhs_format = ValueFormat(prepared, instruction.node.lhs);
  const auto rhs_format = ValueFormat(prepared, instruction.node.rhs);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto lhs = AlignWideFraction(values.wide(instruction.node.lhs, lane),
                                       lhs_format.fraction_bits, fraction);
    const auto rhs = AlignWideFraction(values.wide(instruction.node.rhs, lane),
                                       rhs_format.fraction_bits, fraction);
    result[lane] = lhs < rhs ? lhs : rhs;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMax(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values[instruction.value_index] = ValueMax(
        prepared, values[instruction.node.lhs], values[instruction.node.rhs]);
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const auto lhs_format = ValueFormat(prepared, instruction.node.lhs);
  const auto rhs_format = ValueFormat(prepared, instruction.node.rhs);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto lhs = AlignWideFraction(values.wide(instruction.node.lhs, lane),
                                       lhs_format.fraction_bits, fraction);
    const auto rhs = AlignWideFraction(values.wide(instruction.node.rhs, lane),
                                       rhs_format.fraction_bits, fraction);
    result[lane] = lhs > rhs ? lhs : rhs;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMinUnsigned(const Instruction &instruction,
                               const PreparedRun &, const CpuSimdBindingView &,
                               u64, std::size_t, Values &values) noexcept {
  const Vec lhs = values[instruction.node.lhs];
  const Vec rhs = values[instruction.node.rhs];
  values[instruction.value_index] =
      RUND_CPU_SIMD_VALUE_SELECT(RUND_CPU_SIMD_LT_UNSIGNED(lhs, rhs), lhs, rhs);
}

inline void ExecuteMaxUnsigned(const Instruction &instruction,
                               const PreparedRun &, const CpuSimdBindingView &,
                               u64, std::size_t, Values &values) noexcept {
  const Vec lhs = values[instruction.node.lhs];
  const Vec rhs = values[instruction.node.rhs];
  values[instruction.value_index] =
      RUND_CPU_SIMD_VALUE_SELECT(RUND_CPU_SIMD_GT_UNSIGNED(lhs, rhs), lhs, rhs);
}

inline void ExecuteClamp(const Instruction &instruction,
                         const PreparedRun &prepared,
                         const CpuSimdBindingView &, u64, std::size_t,
                         Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    const Vec lower = ValueMax(prepared, values[instruction.node.lhs],
                               values[instruction.node.rhs]);
    values[instruction.value_index] =
        ValueMin(prepared, lower, values[instruction.node.aux]);
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const auto lhs_format = ValueFormat(prepared, instruction.node.lhs);
  const auto rhs_format = ValueFormat(prepared, instruction.node.rhs);
  const auto aux_format = ValueFormat(prepared, instruction.node.aux);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto value =
        AlignWideFraction(values.wide(instruction.node.lhs, lane),
                          lhs_format.fraction_bits, fraction);
    const auto low = AlignWideFraction(values.wide(instruction.node.rhs, lane),
                                       rhs_format.fraction_bits, fraction);
    const auto high = AlignWideFraction(values.wide(instruction.node.aux, lane),
                                        aux_format.fraction_bits, fraction);
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
  values[instruction.value_index] = RUND_CPU_SIMD_VALUE_SELECT(
      RUND_CPU_SIMD_LT_UNSIGNED(lower, high), lower, high);
}

inline void ExecuteSelect(const Instruction &instruction,
                          const PreparedRun &prepared,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values[instruction.value_index] = RUND_CPU_SIMD_VALUE_SELECT(
        Truthy(values[instruction.node.lhs]), values[instruction.node.rhs],
        values[instruction.node.aux]);
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  const auto true_format = ValueFormat(prepared, instruction.node.rhs);
  const auto false_format = ValueFormat(prepared, instruction.node.aux);
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const auto when_true =
        AlignWideFraction(values.wide(instruction.node.rhs, lane),
                          true_format.fraction_bits, fraction);
    const auto when_false =
        AlignWideFraction(values.wide(instruction.node.aux, lane),
                          false_format.fraction_bits, fraction);
    result[lane] =
        values.wide(instruction.node.lhs, lane) != 0 ? when_true : when_false;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteNeg(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values[instruction.value_index] = RUND_CPU_SIMD_SUB_WRAP(
        RUND_CPU_SIMD_SPLAT(0), values[instruction.node.lhs]);
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    result[lane] = -values.wide(instruction.node.lhs, lane);
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteAbs(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values[instruction.value_index] =
        RUND_CPU_SIMD_ABS(values[instruction.node.lhs]);
    return;
  }
  std::array<WideScalar, kLaneCount> result{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar value = values.wide(instruction.node.lhs, lane);
    result[lane] = value < 0 ? -value : value;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteAbsMagnitude(const Instruction &instruction,
                                const PreparedRun &prepared,
                                const CpuSimdBindingView &, u64, std::size_t,
                                Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    std::array<WideScalar, kLaneCount> result{};
    for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
      const WideScalar value = values.wide(instruction.node.lhs, lane);
      result[lane] = value < 0 ? -value : value;
    }
    values.set_wide(instruction.value_index, result);
    return;
  }
  values[instruction.value_index] =
      SignedBits(RUND_CPU_SIMD_ABS_MAGNITUDE(values[instruction.node.lhs]));
}

inline void ExecuteSign(const Instruction &instruction,
                        const PreparedRun &prepared, const CpuSimdBindingView &,
                        u64, std::size_t, Values &values) noexcept {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed) {
    std::array<WideScalar, kLaneCount> result{};
    for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
      const WideScalar value = values.wide(instruction.node.lhs, lane);
      result[lane] = value < 0   ? WideScalar{-1}
                     : value > 0 ? WideScalar{1}
                                 : WideScalar{0};
    }
    values.set_wide(instruction.value_index, result);
    return;
  }
  values[instruction.value_index] =
      RUND_CPU_SIMD_SIGN(values[instruction.node.lhs]);
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
