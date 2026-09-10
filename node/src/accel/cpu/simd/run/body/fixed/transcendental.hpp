#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

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
