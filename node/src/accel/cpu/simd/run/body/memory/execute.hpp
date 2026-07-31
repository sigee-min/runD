#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

inline void ExecuteParam(const Instruction &instruction, const PreparedRun &,
                         const CpuSimdBindingView &, u64, std::size_t,
                         Values &values) noexcept {
  values[instruction.value_index] = LoadParam(instruction);
}

inline void ExecuteRead(const Instruction &instruction, const PreparedRun &,
                        const CpuSimdBindingView &bindings, const u64 base_tile,
                        const std::size_t live_lanes, Values &values) noexcept {
  values[instruction.value_index] =
      LoadRead(instruction, bindings, base_tile, live_lanes);
}

inline void ExecuteReadUniform(const Instruction &instruction,
                               const PreparedRun &,
                               const CpuSimdBindingView &bindings, u64,
                               std::size_t, Values &values) noexcept {
  values[instruction.value_index] = LoadReadUniform(instruction, bindings);
}

inline void ExecuteReadAt(const Instruction &instruction, const PreparedRun &,
                          const CpuSimdBindingView &bindings,
                          const u64 base_tile, const std::size_t live_lanes,
                          Values &values) noexcept {
  values[instruction.value_index] =
      LoadReadAt(instruction, bindings, base_tile, live_lanes);
}

inline void ExecuteReadFull(const Instruction &instruction, const PreparedRun &,
                            const CpuSimdBindingView &bindings,
                            const u64 base_tile, std::size_t,
                            Values &values) noexcept {
  values[instruction.value_index] =
      LoadReadFull(instruction, bindings, base_tile);
}

inline void ExecuteReadStridedFull(const Instruction &instruction,
                                   const PreparedRun &,
                                   const CpuSimdBindingView &bindings,
                                   const u64 base_tile, std::size_t,
                                   Values &values) noexcept {
  values[instruction.value_index] =
      LoadReadStridedFull(instruction, bindings, base_tile);
}

inline void ExecuteConstant(const Instruction &instruction, const PreparedRun &,
                            const CpuSimdBindingView &, u64, std::size_t,
                            Values &values) noexcept {
  values[instruction.value_index] = RUND_CPU_SIMD_CONSTANT(instruction.node);
}

inline void ExecuteIndex(const Instruction &instruction, const PreparedRun &,
                         const CpuSimdBindingView &bindings,
                         const u64 base_tile, const std::size_t live_lanes,
                         Values &values) noexcept {
  std::array<Scalar, kLaneCount> lanes{};
  for (std::size_t lane = 0u; lane < live_lanes; ++lane) {
    const BitsScalar bits = static_cast<BitsScalar>(
        bindings.logical_offset + base_tile + static_cast<u64>(lane));
    lanes[lane] = std::bit_cast<Scalar>(bits);
  }
  values[instruction.value_index] = RUND_CPU_SIMD_LOAD(lanes.data());
}

inline void ExecuteWrite(const Instruction &instruction, const PreparedRun &,
                         const CpuSimdBindingView &bindings,
                         const u64 base_tile, const std::size_t live_lanes,
                         Values &values) noexcept {
  StoreWrite(instruction, bindings, values[instruction.node.lhs], base_tile,
             live_lanes);
}

inline void ExecuteWriteFull(const Instruction &instruction,
                             const PreparedRun &,
                             const CpuSimdBindingView &bindings,
                             const u64 base_tile, std::size_t,
                             Values &values) noexcept {
  StoreWriteFull(instruction, bindings, values[instruction.node.lhs],
                 base_tile);
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
