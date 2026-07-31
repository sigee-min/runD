#pragma once

#include <math64/simd/mask.hpp>
#include <math64/simd/memory.hpp>
#include <math64/soa/span.hpp>

#include <array>

namespace rund::math64::soa::detail {

[[nodiscard]] inline simd::I64x LoadTailI64(const I64View values,
                                            const std::size_t base,
                                            const i64 pad = 0) noexcept {
  alignas(16) std::array<i64, simd::LaneCount> lanes{pad, pad};
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = values[base + lane];
  }
  return simd::LoadI64(lanes.data());
}

[[nodiscard]] inline simd::I64x LoadTailI64Or(const I64View values,
                                              const std::size_t base,
                                              const simd::I64x fill) noexcept {
  alignas(16) std::array<i64, simd::LaneCount> lanes{};
  simd::Store(lanes.data(), fill);
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = values[base + lane];
  }
  return simd::LoadI64(lanes.data());
}

[[nodiscard]] inline simd::Mask64x TailMask64(const std::size_t size,
                                              const std::size_t base) noexcept {
  return simd::MaskFromBools(base < size, base + 1u < size);
}

inline void StoreTailI64(const I64MutView out, const std::size_t base,
                         const simd::I64x value) noexcept {
  alignas(16) std::array<i64, simd::LaneCount> lanes{};
  simd::Store(lanes.data(), value);
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < out.size(); ++lane) {
    out[base + lane] = lanes[lane];
  }
}

[[nodiscard]] inline simd::I64x
LoadTailI8AsI64(const I8View values, const std::size_t base) noexcept {
  alignas(16) std::array<i64, simd::LaneCount> lanes{};
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = values[base + lane];
  }
  return simd::LoadI64(lanes.data());
}

[[nodiscard]] inline simd::I64x
LoadTailU8AsI64(const U8View values, const std::size_t base) noexcept {
  alignas(16) std::array<i64, simd::LaneCount> lanes{};
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = static_cast<i64>(values[base + lane]);
  }
  return simd::LoadI64(lanes.data());
}

[[nodiscard]] inline simd::I64x
LoadTailI16AsI64(const I16View values, const std::size_t base) noexcept {
  alignas(16) std::array<i64, simd::LaneCount> lanes{};
  for (std::size_t lane = 0u;
       lane < simd::LaneCount && base + lane < values.size(); ++lane) {
    lanes[lane] = values[base + lane];
  }
  return simd::LoadI64(lanes.data());
}

} // namespace rund::math64::soa::detail
