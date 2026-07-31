#pragma once

#include <math32/simd/mask.hpp>
#include <math32/simd/memory.hpp>
#include <math32/soa/span.hpp>

#include <array>

namespace rund::math32::soa::detail {

[[nodiscard]] inline simd::I32x LoadTailI32(const I32View values,
                                            const std::size_t base,
                                            const i32 pad = 0) noexcept {
  alignas(16) std::array<i32, simd::LaneCount> lanes{pad, pad, pad, pad};
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = values[base + lane];
  }
  return simd::LoadI32(lanes.data());
}

[[nodiscard]] inline simd::I32x LoadTailI32Or(const I32View values,
                                              const std::size_t base,
                                              const simd::I32x fill) noexcept {
  alignas(16) std::array<i32, simd::LaneCount> lanes{};
  simd::Store(lanes.data(), fill);
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = values[base + lane];
  }
  return simd::LoadI32(lanes.data());
}

[[nodiscard]] inline simd::Mask32x TailMask32(const std::size_t size,
                                              const std::size_t base) noexcept {
  return simd::MaskFromBools(base < size, base + 1u < size, base + 2u < size,
                             base + 3u < size);
}

inline void StoreTailI32(const I32MutView out, const std::size_t base,
                         const simd::I32x value) noexcept {
  alignas(16) std::array<i32, simd::LaneCount> lanes{};
  simd::Store(lanes.data(), value);
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < out.size(); ++lane) {
    out[base + lane] = lanes[lane];
  }
}

[[nodiscard]] inline simd::I32x
LoadTailI8AsI32(const I8View values, const std::size_t base) noexcept {
  alignas(16) std::array<i32, simd::LaneCount> lanes{};
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = values[base + lane];
  }
  return simd::LoadI32(lanes.data());
}

[[nodiscard]] inline simd::I32x
LoadTailU8AsI32(const U8View values, const std::size_t base) noexcept {
  alignas(16) std::array<i32, simd::LaneCount> lanes{};
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = static_cast<i32>(values[base + lane]);
  }
  return simd::LoadI32(lanes.data());
}

[[nodiscard]] inline simd::I32x
LoadTailI16AsI32(const I16View values, const std::size_t base) noexcept {
  alignas(16) std::array<i32, simd::LaneCount> lanes{};
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = values[base + lane];
  }
  return simd::LoadI32(lanes.data());
}

} // namespace rund::math32::soa::detail
