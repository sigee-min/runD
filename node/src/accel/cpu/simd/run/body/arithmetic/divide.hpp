#pragma once

inline void ExecuteDivSigned(const Instruction &instruction,
                             const PreparedRun &, const CpuSimdBindingView &,
                             u64, const std::size_t live_lanes,
                             Values &values) noexcept {
  const auto left = std::bit_cast<std::array<Scalar, kLaneCount>>(
      values[instruction.node.lhs]);
  const auto right = std::bit_cast<std::array<Scalar, kLaneCount>>(
      values[instruction.node.rhs]);
  std::array<Scalar, kLaneCount> output{};
  for (std::size_t lane = 0u; lane < live_lanes; ++lane) {
    if (right[lane] == Scalar{0}) {
      values.fail("compute_integer_divide_by_zero");
      return;
    }
    if (left[lane] == std::numeric_limits<Scalar>::min() &&
        right[lane] == Scalar{-1}) {
      values.fail("compute_integer_divide_overflow");
      return;
    }
    output[lane] = static_cast<Scalar>(left[lane] / right[lane]);
  }
  values[instruction.value_index] = std::bit_cast<Vec>(output);
}
inline void ExecuteDivUnsigned(const Instruction &instruction,
                               const PreparedRun &, const CpuSimdBindingView &,
                               u64, const std::size_t live_lanes,
                               Values &values) noexcept {
  const auto left = std::bit_cast<std::array<BitsScalar, kLaneCount>>(
      Bits(values[instruction.node.lhs]));
  const auto right = std::bit_cast<std::array<BitsScalar, kLaneCount>>(
      Bits(values[instruction.node.rhs]));
  std::array<BitsScalar, kLaneCount> output{};
  for (std::size_t lane = 0u; lane < live_lanes; ++lane) {
    if (right[lane] == BitsScalar{0}) {
      values.fail("compute_integer_divide_by_zero");
      return;
    }
    output[lane] = static_cast<BitsScalar>(left[lane] / right[lane]);
  }
  values[instruction.value_index] = SignedBits(std::bit_cast<BitsVec>(output));
}
