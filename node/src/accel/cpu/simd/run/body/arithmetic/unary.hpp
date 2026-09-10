#pragma once

inline void ExecuteNeg(const Instruction &instruction,
                       const PreparedRun &prepared, const CpuSimdBindingView &,
                       u64, std::size_t, Values &values) noexcept {
  if (prepared.domain != rund::kernel::ComputeDomain::Fixed) {
    values.set_raw(instruction.value_index,
                   RUND_CPU_SIMD_SUB_WRAP(RUND_CPU_SIMD_SPLAT(0),
                                          values[instruction.node.lhs]));
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
    values.set_raw(instruction.value_index,
                   RUND_CPU_SIMD_ABS(values[instruction.node.lhs]));
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
  values.set_raw(
      instruction.value_index,
      SignedBits(RUND_CPU_SIMD_ABS_MAGNITUDE(values[instruction.node.lhs])));
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
  values.set_raw(instruction.value_index,
                 RUND_CPU_SIMD_SIGN(values[instruction.node.lhs]));
}
