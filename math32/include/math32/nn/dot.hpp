#pragma once

#include <math32/nn/model.hpp>
#include <math32/soa/span.hpp>
#include <math32/soa/tail.hpp>

namespace rund::math32::nn {
struct DotAccumulator {
  simd::I32x sum{};
  u32 processed = 0u;
  bool size_match = true;
  bool empty_input = false;
  [[nodiscard]] constexpr bool ok() const noexcept { return size_match; }
};
namespace detail {
using I8x4 = i8 __attribute__((vector_size(4)));
using U8x4 = u8 __attribute__((vector_size(4)));
using I16x4 = i16 __attribute__((vector_size(8)));
inline I8x4 LoadI8x4(const i8* source) noexcept {
  I8x4 value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}
inline U8x4 LoadU8x4(const u8* source) noexcept {
  U8x4 value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}
inline I16x4 LoadI16x4(const i16* source) noexcept {
  I16x4 value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}
}  // namespace detail
[[nodiscard]] inline DotAccumulator DotI8I8To32(const soa::I8View lhs, const soa::I8View rhs) noexcept {
  if (lhs.size() != rhs.size()) return DotAccumulator{.sum = simd::SplatI32(0), .processed = 0u, .size_match = false};
  if (lhs.empty()) return DotAccumulator{.sum = simd::SplatI32(0), .processed = 0u, .empty_input = true};
  simd::I32x sum = simd::SplatI32(0);
  u32 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= lhs.size(); index += simd::LaneCount) {
    sum = AddWrap(sum, MulLow(__builtin_convertvector(detail::LoadI8x4(lhs.data() + index), simd::I32x),
                              __builtin_convertvector(detail::LoadI8x4(rhs.data() + index), simd::I32x)));
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (index < lhs.size()) {
    sum = AddWrap(sum, MulLow(soa::detail::LoadTailI8AsI32(lhs, index), soa::detail::LoadTailI8AsI32(rhs, index)));
    processed += ::rund::math32::detail::ScalarSatU32(lhs.size() - index);
  }
  return DotAccumulator{.sum = sum, .processed = processed};
}
[[nodiscard]] inline DotAccumulator DotU8I8To32(const soa::U8View lhs, const soa::I8View rhs) noexcept {
  if (lhs.size() != rhs.size()) return DotAccumulator{.sum = simd::SplatI32(0), .processed = 0u, .size_match = false};
  if (lhs.empty()) return DotAccumulator{.sum = simd::SplatI32(0), .processed = 0u, .empty_input = true};
  simd::I32x sum = simd::SplatI32(0);
  u32 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= lhs.size(); index += simd::LaneCount) {
    sum = AddWrap(sum, MulLow(__builtin_convertvector(detail::LoadU8x4(lhs.data() + index), simd::I32x),
                              __builtin_convertvector(detail::LoadI8x4(rhs.data() + index), simd::I32x)));
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (index < lhs.size()) {
    sum = AddWrap(sum, MulLow(soa::detail::LoadTailU8AsI32(lhs, index), soa::detail::LoadTailI8AsI32(rhs, index)));
    processed += ::rund::math32::detail::ScalarSatU32(lhs.size() - index);
  }
  return DotAccumulator{.sum = sum, .processed = processed};
}
[[nodiscard]] inline DotAccumulator DotI16I16To32(const soa::I16View lhs, const soa::I16View rhs) noexcept {
  if (lhs.size() != rhs.size()) return DotAccumulator{.sum = simd::SplatI32(0), .processed = 0u, .size_match = false};
  if (lhs.empty()) return DotAccumulator{.sum = simd::SplatI32(0), .processed = 0u, .empty_input = true};
  simd::I32x sum = simd::SplatI32(0);
  u32 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= lhs.size(); index += simd::LaneCount) {
    sum = AddWrap(sum, MulLow(__builtin_convertvector(detail::LoadI16x4(lhs.data() + index), simd::I32x),
                              __builtin_convertvector(detail::LoadI16x4(rhs.data() + index), simd::I32x)));
    processed += static_cast<u32>(simd::LaneCount);
  }
  if (index < lhs.size()) {
    sum = AddWrap(sum, MulLow(soa::detail::LoadTailI16AsI32(lhs, index), soa::detail::LoadTailI16AsI32(rhs, index)));
    processed += ::rund::math32::detail::ScalarSatU32(lhs.size() - index);
  }
  return DotAccumulator{.sum = sum, .processed = processed};
}
}  // namespace rund::math32::nn
