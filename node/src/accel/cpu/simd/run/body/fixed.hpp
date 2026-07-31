#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

using UnsignedWideScalar = __uint128_t;

[[nodiscard]] inline UnsignedWideScalar
WideMagnitude(const WideScalar value) noexcept {
  return value < 0 ? static_cast<UnsignedWideScalar>(-(value + 1)) + 1u
                   : static_cast<UnsignedWideScalar>(value);
}

[[nodiscard]] inline unsigned
StoredWidth(const rund::kernel::ComputeFixedFormat format) noexcept {
  return static_cast<unsigned>(format.integer_bits) + format.fraction_bits;
}

[[nodiscard]] inline UnsignedWideScalar
StoredUnsignedBits(const WideScalar value,
                   const rund::kernel::ComputeFixedFormat format) noexcept {
  const unsigned width = StoredWidth(format);
  const UnsignedWideScalar mask = width == 128u
                                      ? ~UnsignedWideScalar{0u}
                                      : (UnsignedWideScalar{1u} << width) - 1u;
  return static_cast<UnsignedWideScalar>(value) & mask;
}

[[nodiscard]] inline WideScalar
StoredSignedBits(UnsignedWideScalar value,
                 const rund::kernel::ComputeFixedFormat format) noexcept {
  const unsigned width = StoredWidth(format);
  const UnsignedWideScalar sign = UnsignedWideScalar{1u} << (width - 1u);
  const UnsignedWideScalar mask = width == 128u
                                      ? ~UnsignedWideScalar{0u}
                                      : (UnsignedWideScalar{1u} << width) - 1u;
  value &= mask;
  if (width < 128u && (value & sign) != 0u) {
    value |= ~mask;
  }
  return SignedWideFromBits(value);
}

[[nodiscard]] inline rund::kernel::ComputeFixedFormat
FixedProductFormat(const rund::kernel::ComputeFixedFormat format) noexcept {
  auto product = format;
  product.integer_bits = static_cast<rund::kernel::u8>(
      static_cast<unsigned>(format.integer_bits) * 2u);
  product.fraction_bits = static_cast<rund::kernel::u8>(
      static_cast<unsigned>(format.fraction_bits) * 2u);
  return product;
}

[[nodiscard]] inline WideScalar QuantizeUnsignedFixedProduct(
    const UnsignedWideScalar product,
    const rund::kernel::ComputeFixedFormat format) noexcept {
  const unsigned shift = format.fraction_bits;
  UnsignedWideScalar quotient = product >> shift;
  const UnsignedWideScalar mask = (UnsignedWideScalar{1u} << shift) - 1u;
  const UnsignedWideScalar remainder = product & mask;
  const UnsignedWideScalar halfway = UnsignedWideScalar{1u} << (shift - 1u);
  const bool nonzero = remainder != 0u;
  const bool nearest =
      remainder > halfway || (remainder == halfway && (quotient & 1u) != 0u);
  if ((format.rounding == rund::kernel::ComputeRounding::Up && nonzero) ||
      (format.rounding == rund::kernel::ComputeRounding::NearestEven &&
       nearest)) {
    ++quotient;
  }
  const unsigned width = StoredWidth(format);
  const UnsignedWideScalar maximum = (UnsignedWideScalar{1u} << width) - 1u;
  if (format.overflow == rund::kernel::ComputeOverflow::Saturate &&
      quotient > maximum) {
    quotient = maximum;
  }
  return StoredSignedBits(quotient & maximum, format);
}

[[nodiscard]] inline WideScalar
FixedDivideRaw(const WideScalar lhs, const WideScalar rhs,
               const rund::kernel::ComputeFixedFormat format) noexcept {
  const unsigned fraction = format.fraction_bits;
  if (rhs == 0) {
    if (lhs == 0) {
      return 0;
    }
    const UnsignedWideScalar sign = UnsignedWideScalar{1u}
                                    << (format.integer_bits + fraction - 1u);
    return lhs < 0 ? -static_cast<WideScalar>(sign)
                   : static_cast<WideScalar>(sign - 1u);
  }
  const bool negative = (lhs < 0) != (rhs < 0);
  const UnsignedWideScalar numerator = WideMagnitude(lhs) << fraction;
  const UnsignedWideScalar denominator = WideMagnitude(rhs);
  UnsignedWideScalar quotient = numerator / denominator;
  const UnsignedWideScalar remainder = numerator % denominator;
  const bool nonzero = remainder != 0u;
  const UnsignedWideScalar twice = remainder << 1u;
  const bool nearest =
      twice > denominator || (twice == denominator && (quotient & 1u) != 0u);
  if ((format.rounding == rund::kernel::ComputeRounding::Down && negative &&
       nonzero) ||
      (format.rounding == rund::kernel::ComputeRounding::Up && !negative &&
       nonzero) ||
      (format.rounding == rund::kernel::ComputeRounding::NearestEven &&
       nearest)) {
    ++quotient;
  }
  return negative ? -static_cast<WideScalar>(quotient)
                  : static_cast<WideScalar>(quotient);
}

[[nodiscard]] inline WideScalar
FixedSqrtRaw(const WideScalar value,
             const rund::kernel::ComputeFixedFormat format) noexcept {
  if (value <= 0) {
    return 0;
  }
  const UnsignedWideScalar radicand = static_cast<UnsignedWideScalar>(value)
                                      << format.fraction_bits;
  UnsignedWideScalar root = 0u;
  for (unsigned step = 0u; step < 64u; ++step) {
    const UnsignedWideScalar candidate =
        root | (UnsignedWideScalar{1u} << (63u - step));
    if (candidate * candidate <= radicand) {
      root = candidate;
    }
  }
  return static_cast<WideScalar>(root);
}

template <class Operation>
inline void ExecuteFixedWideUnary(const Instruction &instruction,
                                  Values &values,
                                  Operation &&operation) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar raw = operation(values.wide(instruction.node.lhs, lane),
                                     instruction.node.fixed_format);
    result[lane] = QuantizeWide(raw, instruction.node.fixed_format,
                                instruction.node.fixed_format);
  }
  values.set_wide(instruction.value_index, result);
}

[[nodiscard]] inline rund::kernel::ComputeFixedFormat
CanonicalUnitFormat(const rund::kernel::ComputeFixedFormat format) noexcept {
  return rund::kernel::ComputeFixedFormat{
      .integer_bits = 1u,
      .fraction_bits = static_cast<rund::kernel::u8>(RUND_CPU_SIMD_BITS - 1u),
      .rounding = format.rounding,
      .overflow = rund::kernel::ComputeOverflow::Saturate,
      .approximation = format.approximation};
}

template <class Input, class Operation>
inline void
ExecuteFixedCanonicalUnary(const Instruction &instruction, Values &values,
                           Input &&input, Operation &&operation,
                           const unsigned output_fraction) noexcept {
  std::array<RUND_CPU_SIMD_SCALAR, kLaneCount> canonical_input{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    canonical_input[lane] = input(values.wide(instruction.node.lhs, lane),
                                  instruction.node.fixed_format);
  }
  const RUND_CPU_SIMD_VEC canonical_output =
      operation(RUND_CPU_SIMD_LOAD(canonical_input.data()));
  std::array<RUND_CPU_SIMD_SCALAR, kLaneCount> raw_output{};
  RUND_CPU_SIMD_STORE(raw_output.data(), canonical_output);
  auto source = instruction.node.fixed_format;
  source.integer_bits =
      static_cast<rund::kernel::u8>(RUND_CPU_SIMD_BITS - output_fraction);
  source.fraction_bits = static_cast<rund::kernel::u8>(output_fraction);
  std::array<WideScalar, kLaneCount> result{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    result[lane] = QuantizeWide(static_cast<WideScalar>(raw_output[lane]),
                                source, instruction.node.fixed_format);
  }
  values.set_wide(instruction.value_index, result);
}

[[nodiscard]] inline RUND_CPU_SIMD_SCALAR
FixedTurnPhase(const WideScalar value,
               const rund::kernel::ComputeFixedFormat format) noexcept {
  const auto bits = static_cast<RUND_CPU_SIMD_BITS_SCALAR>(value);
  const auto phase = static_cast<RUND_CPU_SIMD_BITS_SCALAR>(
      bits << (RUND_CPU_SIMD_BITS - format.fraction_bits));
  return std::bit_cast<RUND_CPU_SIMD_SCALAR>(phase);
}

[[nodiscard]] inline RUND_CPU_SIMD_SCALAR
FixedCanonicalUnit(const WideScalar value,
                   const rund::kernel::ComputeFixedFormat format) noexcept {
  return static_cast<RUND_CPU_SIMD_SCALAR>(
      QuantizeWide(value, format, CanonicalUnitFormat(format)));
}

inline void ExecuteAddSat(const Instruction &instruction, const PreparedRun &,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  values[instruction.value_index] = RUND_CPU_SIMD_ADD_SAT(
      values[instruction.node.lhs], values[instruction.node.rhs]);
}

inline void ExecuteAddSatUnsigned(const Instruction &instruction,
                                  const PreparedRun &,
                                  const CpuSimdBindingView &, u64, std::size_t,
                                  Values &values) noexcept {
  values[instruction.value_index] = SignedBits(RUND_CPU_SIMD_ADD_SAT_UNSIGNED(
      Bits(values[instruction.node.lhs]), Bits(values[instruction.node.rhs])));
}

inline void ExecuteSubSat(const Instruction &instruction, const PreparedRun &,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  values[instruction.value_index] = RUND_CPU_SIMD_SUB_SAT(
      values[instruction.node.lhs], values[instruction.node.rhs]);
}

inline void ExecuteNegPositiveFixed(const Instruction &instruction,
                                    const PreparedRun &,
                                    const CpuSimdBindingView &, u64,
                                    std::size_t, Values &values) noexcept {
  values[instruction.value_index] =
      RUND_CPU_SIMD_NEG_POSITIVE_FIXED(values[instruction.node.lhs]);
}

inline void ExecuteMulFixed(const Instruction &instruction, const PreparedRun &,
                            const CpuSimdBindingView &, u64, std::size_t,
                            Values &values) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  const auto source = FixedProductFormat(instruction.node.fixed_format);
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar product = values.wide(instruction.node.lhs, lane) *
                               values.wide(instruction.node.rhs, lane);
    result[lane] = QuantizeWide(product, source, instruction.node.fixed_format);
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMulFixedScaled(const Instruction &instruction,
                                  const PreparedRun &,
                                  const CpuSimdBindingView &, u64, std::size_t,
                                  Values &values) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  const auto format = instruction.node.fixed_format;
  const auto source = FixedProductFormat(format);
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar coefficient = static_cast<WideScalar>(
        StoredUnsignedBits(values.wide(instruction.node.rhs, lane), format));
    const WideScalar product =
        values.wide(instruction.node.lhs, lane) * coefficient;
    result[lane] = QuantizeWide(product, source, format);
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMulUnsignedFixed(const Instruction &instruction,
                                    const PreparedRun &,
                                    const CpuSimdBindingView &, u64,
                                    std::size_t, Values &values) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  const auto format = instruction.node.fixed_format;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const UnsignedWideScalar lhs =
        StoredUnsignedBits(values.wide(instruction.node.lhs, lane), format);
    const UnsignedWideScalar rhs =
        StoredUnsignedBits(values.wide(instruction.node.rhs, lane), format);
    result[lane] = QuantizeUnsignedFixedProduct(lhs * rhs, format);
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteMulAddFixed(const Instruction &instruction,
                               const PreparedRun &prepared,
                               const CpuSimdBindingView &, u64, std::size_t,
                               Values &values) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  const auto lhs_format = ValueFormat(prepared, instruction.node.lhs);
  const auto rhs_format = ValueFormat(prepared, instruction.node.rhs);
  const auto addend_format = ValueFormat(prepared, instruction.node.aux);
  const unsigned product_fraction =
      static_cast<unsigned>(lhs_format.fraction_bits) +
      rhs_format.fraction_bits;
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar product = values.wide(instruction.node.lhs, lane) *
                               values.wide(instruction.node.rhs, lane);
    const WideScalar aligned_product =
        AlignWideFraction(product, product_fraction, fraction);
    const WideScalar aligned_addend =
        AlignWideFraction(values.wide(instruction.node.aux, lane),
                          addend_format.fraction_bits, fraction);
    result[lane] = aligned_product + aligned_addend;
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteDivFixed(const Instruction &instruction, const PreparedRun &,
                            const CpuSimdBindingView &, u64, std::size_t,
                            Values &values) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar raw = FixedDivideRaw(
        values.wide(instruction.node.lhs, lane),
        values.wide(instruction.node.rhs, lane), instruction.node.fixed_format);
    result[lane] = QuantizeWide(raw, instruction.node.fixed_format,
                                instruction.node.fixed_format);
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteRecip(const Instruction &instruction, const PreparedRun &,
                         const CpuSimdBindingView &, u64, std::size_t,
                         Values &values) noexcept {
  ExecuteFixedWideUnary(
      instruction, values,
      [](const WideScalar value,
         const rund::kernel::ComputeFixedFormat format) noexcept {
        const WideScalar one = WideScalar{1} << format.fraction_bits;
        return FixedDivideRaw(one, value, format);
      });
}

inline void ExecuteSqrt(const Instruction &instruction, const PreparedRun &,
                        const CpuSimdBindingView &, u64, std::size_t,
                        Values &values) noexcept {
  ExecuteFixedWideUnary(instruction, values, FixedSqrtRaw);
}

inline void ExecuteRsqrt(const Instruction &instruction, const PreparedRun &,
                         const CpuSimdBindingView &, u64, std::size_t,
                         Values &values) noexcept {
  ExecuteFixedWideUnary(
      instruction, values,
      [](const WideScalar value,
         const rund::kernel::ComputeFixedFormat format) noexcept {
        const WideScalar root = FixedSqrtRaw(value, format);
        const WideScalar one = WideScalar{1} << format.fraction_bits;
        return FixedDivideRaw(one, root, format);
      });
}

inline void ExecuteSin(const Instruction &instruction, const PreparedRun &,
                       const CpuSimdBindingView &, u64, std::size_t,
                       Values &values) noexcept {
  ExecuteFixedCanonicalUnary(
      instruction, values, FixedTurnPhase,
      [](const RUND_CPU_SIMD_VEC value) noexcept {
        return RUND_CPU_SIMD_SIN(value);
      },
      RUND_CPU_SIMD_BITS - 1u);
}

inline void ExecuteCos(const Instruction &instruction, const PreparedRun &,
                       const CpuSimdBindingView &, u64, std::size_t,
                       Values &values) noexcept {
  ExecuteFixedCanonicalUnary(
      instruction, values, FixedTurnPhase,
      [](const RUND_CPU_SIMD_VEC value) noexcept {
        return RUND_CPU_SIMD_COS(value);
      },
      RUND_CPU_SIMD_BITS - 1u);
}

inline void ExecuteTan(const Instruction &instruction, const PreparedRun &,
                       const CpuSimdBindingView &, u64, std::size_t,
                       Values &values) noexcept {
  std::array<RUND_CPU_SIMD_SCALAR, kLaneCount> phase{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    phase[lane] = FixedTurnPhase(values.wide(instruction.node.lhs, lane),
                                 instruction.node.fixed_format);
  }
  const RUND_CPU_SIMD_VEC input = RUND_CPU_SIMD_LOAD(phase.data());
  const RUND_CPU_SIMD_VEC sin_value = RUND_CPU_SIMD_SIN(input);
  const RUND_CPU_SIMD_VEC cos_value = RUND_CPU_SIMD_COS(input);
  std::array<RUND_CPU_SIMD_SCALAR, kLaneCount> sin_raw{};
  std::array<RUND_CPU_SIMD_SCALAR, kLaneCount> cos_raw{};
  RUND_CPU_SIMD_STORE(sin_raw.data(), sin_value);
  RUND_CPU_SIMD_STORE(cos_raw.data(), cos_value);
  std::array<WideScalar, kLaneCount> result{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar quotient = FixedDivideRaw(sin_raw[lane], cos_raw[lane],
                                               instruction.node.fixed_format);
    result[lane] = QuantizeWide(quotient, instruction.node.fixed_format,
                                instruction.node.fixed_format);
  }
  values.set_wide(instruction.value_index, result);
}

inline void ExecuteExp(const Instruction &instruction, const PreparedRun &,
                       const CpuSimdBindingView &, u64, std::size_t,
                       Values &values) noexcept {
  ExecuteFixedCanonicalUnary(
      instruction, values, FixedCanonicalUnit,
      [](const RUND_CPU_SIMD_VEC value) noexcept {
        return RUND_CPU_SIMD_EXP(value);
      },
      RUND_CPU_SIMD_BITS - 1u);
}

inline void ExecuteLog(const Instruction &instruction, const PreparedRun &,
                       const CpuSimdBindingView &, u64, std::size_t,
                       Values &values) noexcept {
  ExecuteFixedCanonicalUnary(
      instruction, values, FixedCanonicalUnit,
      [](const RUND_CPU_SIMD_VEC value) noexcept {
        return RUND_CPU_SIMD_LOG(value);
      },
      RUND_CPU_SIMD_BITS - 1u);
}

inline void ExecuteAtan2(const Instruction &instruction, const PreparedRun &,
                         const CpuSimdBindingView &, u64, std::size_t,
                         Values &values) noexcept {
  const RUND_CPU_SIMD_VEC phase = RUND_CPU_SIMD_ATAN2(
      values[instruction.node.lhs], values[instruction.node.rhs]);
  std::array<RUND_CPU_SIMD_SCALAR, kLaneCount> raw{};
  RUND_CPU_SIMD_STORE(raw.data(), phase);
  auto source = instruction.node.fixed_format;
  source.integer_bits = 0u;
  source.fraction_bits = static_cast<rund::kernel::u8>(RUND_CPU_SIMD_BITS);
  std::array<WideScalar, kLaneCount> result{};
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    result[lane] = QuantizeWide(static_cast<WideScalar>(raw[lane]), source,
                                instruction.node.fixed_format);
  }
  values.set_wide(instruction.value_index, result);
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
