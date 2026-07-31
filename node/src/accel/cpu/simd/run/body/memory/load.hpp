#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] Vec LoadParam(const Instruction &instruction) noexcept {
  Scalar value{};
  std::memcpy(&value, &instruction.immediate, sizeof(value));
  return RUND_CPU_SIMD_SPLAT(value);
}

[[nodiscard]] Vec LoadRead(const Instruction &instruction,
                           const CpuSimdBindingView &bindings,
                           const u64 base_tile,
                           const std::size_t live_lanes) noexcept {
  std::array<Scalar, kLaneCount> lanes{};
  const auto &binding = bindings.reads[instruction.binding_slot];
  const auto *const read = binding.data;
  const auto stride = binding.stride;
  const unsigned char *cursor = read + ByteOffset(base_tile, stride);
  for (std::size_t lane = 0u; lane < live_lanes; ++lane) {
    std::memcpy(&lanes[lane], cursor, sizeof(Scalar));
    cursor += stride;
  }
  return RUND_CPU_SIMD_LOAD(lanes.data());
}

[[nodiscard]] Vec LoadReadAt(const Instruction &instruction,
                             const CpuSimdBindingView &bindings,
                             const u64 base_tile,
                             const std::size_t live_lanes) noexcept {
  std::array<Scalar, kLaneCount> lanes{};
  const auto &source = bindings.reads[instruction.binding_slot];
  const auto &indices = bindings.reads[instruction.immediate];
  for (std::size_t lane = 0u; lane < live_lanes; ++lane) {
    u32 index = 0u;
    std::memcpy(&index,
                indices.data +
                    ByteOffset(base_tile + static_cast<u64>(lane),
                               indices.stride),
                sizeof(index));
    if (index < instruction.element_bytes) {
      std::memcpy(&lanes[lane],
                  source.data + ByteOffset(index, source.stride),
                  sizeof(Scalar));
    }
  }
  return RUND_CPU_SIMD_LOAD(lanes.data());
}

[[nodiscard]] Vec LoadReadStridedFull(const Instruction &instruction,
                                      const CpuSimdBindingView &bindings,
                                      const u64 base_tile) noexcept {
  std::array<Scalar, kLaneCount> lanes;
  const auto &binding = bindings.reads[instruction.binding_slot];
  const auto *const read = binding.data;
  const auto stride = binding.stride;
  const unsigned char *cursor = read + ByteOffset(base_tile, stride);
  for (std::size_t lane = 0u; lane < kLaneCount; ++lane) {
    std::memcpy(&lanes[lane], cursor, sizeof(Scalar));
    cursor += stride;
  }
  return RUND_CPU_SIMD_LOAD(lanes.data());
}

[[nodiscard]] Vec LoadReadFull(const Instruction &instruction,
                               const CpuSimdBindingView &bindings,
                               const u64 base_tile) noexcept {
  const auto &binding = bindings.reads[instruction.binding_slot];
  const auto *const read = binding.data;
  const auto stride = binding.stride;
  Vec value{};
  std::memcpy(&value, read + ByteOffset(base_tile, stride), sizeof(value));
  return value;
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
