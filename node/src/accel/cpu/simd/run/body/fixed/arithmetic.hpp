#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

inline void ExecuteAddSat(const Instruction &instruction, const PreparedRun &,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  values.set_raw(instruction.value_index,
                 RUND_CPU_SIMD_ADD_SAT(values[instruction.node.lhs],
                                       values[instruction.node.rhs]));
}

inline void ExecuteAddSatUnsigned(const Instruction &instruction,
                                  const PreparedRun &,
                                  const CpuSimdBindingView &, u64, std::size_t,
                                  Values &values) noexcept {
  values.set_raw(instruction.value_index,
                 SignedBits(RUND_CPU_SIMD_ADD_SAT_UNSIGNED(
                     Bits(values[instruction.node.lhs]),
                     Bits(values[instruction.node.rhs]))));
}

inline void ExecuteSubSat(const Instruction &instruction, const PreparedRun &,
                          const CpuSimdBindingView &, u64, std::size_t,
                          Values &values) noexcept {
  values.set_raw(instruction.value_index,
                 RUND_CPU_SIMD_SUB_SAT(values[instruction.node.lhs],
                                       values[instruction.node.rhs]));
}

inline void ExecuteNegPositiveFixed(const Instruction &instruction,
                                    const PreparedRun &,
                                    const CpuSimdBindingView &, u64,
                                    std::size_t, Values &values) noexcept {
  values.set_raw(instruction.value_index, RUND_CPU_SIMD_NEG_POSITIVE_FIXED(
                                              values[instruction.node.lhs]));
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
                               const PreparedRun &, const CpuSimdBindingView &,
                               u64, std::size_t, Values &values) noexcept {
  std::array<WideScalar, kLaneCount> result{};
  const unsigned lhs_fraction =
      ValueFractionBits(instruction, instruction.node.lhs);
  const unsigned rhs_fraction =
      ValueFractionBits(instruction, instruction.node.rhs);
  const unsigned addend_fraction =
      ValueFractionBits(instruction, instruction.node.aux);
  const unsigned product_fraction = lhs_fraction + rhs_fraction;
  const unsigned fraction = instruction.node.fixed_format.fraction_bits;
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    const WideScalar product = values.wide(instruction.node.lhs, lane) *
                               values.wide(instruction.node.rhs, lane);
    const WideScalar aligned_product =
        AlignWideFraction(product, product_fraction, fraction);
    const WideScalar aligned_addend = AlignWideFraction(
        values.wide(instruction.node.aux, lane), addend_fraction, fraction);
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

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
