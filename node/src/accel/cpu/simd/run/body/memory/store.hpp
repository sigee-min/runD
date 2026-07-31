#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

inline void StoreWrite(const Instruction &instruction,
                       const CpuSimdBindingView &bindings, const Vec value,
                       const u64 base_tile,
                       const std::size_t live_lanes) noexcept {
  std::array<Scalar, kLaneCount> lanes;
  RUND_CPU_SIMD_STORE(lanes.data(), value);
  const auto &binding = bindings.writes[instruction.binding_slot];
  auto *const write = binding.data;
  const auto stride = binding.stride;
  unsigned char *cursor = write + ByteOffset(base_tile, stride);
  for (std::size_t lane = 0u; lane < live_lanes; ++lane) {
    if (instruction.element_bytes == sizeof(Scalar)) {
      std::memcpy(cursor, &lanes[lane], sizeof(Scalar));
    } else if (instruction.element_bytes == sizeof(u32)) {
      const u32 low = static_cast<u32>(std::bit_cast<BitsScalar>(lanes[lane]));
      std::memcpy(cursor, &low, sizeof(low));
    } else {
      const u64 widened =
          static_cast<u32>(std::bit_cast<BitsScalar>(lanes[lane]));
      std::memcpy(cursor, &widened, sizeof(widened));
    }
    cursor += stride;
  }
}

inline void StoreWriteFull(const Instruction &instruction,
                           const CpuSimdBindingView &bindings, const Vec value,
                           const u64 base_tile) noexcept {
  const auto &binding = bindings.writes[instruction.binding_slot];
  auto *const write = binding.data;
  const auto stride = binding.stride;
  std::memcpy(write + ByteOffset(base_tile, stride), &value, sizeof(value));
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
