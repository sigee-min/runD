#pragma once

#include <math64/nn/model.hpp>
#include <math64/soa/span.hpp>
#include <math64/soa/tail.hpp>

namespace rund::math64::nn {
struct DotAccumulator {
  simd::I64x sum{};
  u64 processed = 0u;
  bool size_match = true;
  bool empty_input = false;
  [[nodiscard]] constexpr bool ok() const noexcept { return size_match; }
};
namespace detail {
using I8x2 = i8 __attribute__((vector_size(2)));
using U8x2 = u8 __attribute__((vector_size(2)));
using I16x2 = i16 __attribute__((vector_size(4)));
inline I8x2 LoadI8x2(const i8* source) noexcept {
  I8x2 value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}
inline U8x2 LoadU8x2(const u8* source) noexcept {
  U8x2 value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}
inline I16x2 LoadI16x2(const i16* source) noexcept {
  I16x2 value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}
}  // namespace detail
[[nodiscard]] inline DotAccumulator DotI8I8To64(const soa::I8View lhs, const soa::I8View rhs) noexcept {
  if (lhs.size() != rhs.size()) return DotAccumulator{.sum = simd::SplatI64(0), .processed = 0u, .size_match = false};
  if (lhs.empty()) return DotAccumulator{.sum = simd::SplatI64(0), .processed = 0u, .empty_input = true};
  simd::I64x sum = simd::SplatI64(0);
  u64 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= lhs.size(); index += simd::LaneCount) {
    sum = AddWrap(sum, MulLow(__builtin_convertvector(detail::LoadI8x2(lhs.data() + index), simd::I64x),
                              __builtin_convertvector(detail::LoadI8x2(rhs.data() + index), simd::I64x)));
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (index < lhs.size()) {
    sum = AddWrap(sum, MulLow(soa::detail::LoadTailI8AsI64(lhs, index), soa::detail::LoadTailI8AsI64(rhs, index)));
    processed += ::rund::math64::detail::ScalarSatU32(lhs.size() - index);
  }
  return DotAccumulator{.sum = sum, .processed = processed};
}
[[nodiscard]] inline DotAccumulator DotU8I8To64(const soa::U8View lhs, const soa::I8View rhs) noexcept {
  if (lhs.size() != rhs.size()) return DotAccumulator{.sum = simd::SplatI64(0), .processed = 0u, .size_match = false};
  if (lhs.empty()) return DotAccumulator{.sum = simd::SplatI64(0), .processed = 0u, .empty_input = true};
  simd::I64x sum = simd::SplatI64(0);
  u64 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= lhs.size(); index += simd::LaneCount) {
    sum = AddWrap(sum, MulLow(__builtin_convertvector(detail::LoadU8x2(lhs.data() + index), simd::I64x),
                              __builtin_convertvector(detail::LoadI8x2(rhs.data() + index), simd::I64x)));
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (index < lhs.size()) {
    sum = AddWrap(sum, MulLow(soa::detail::LoadTailU8AsI64(lhs, index), soa::detail::LoadTailI8AsI64(rhs, index)));
    processed += ::rund::math64::detail::ScalarSatU32(lhs.size() - index);
  }
  return DotAccumulator{.sum = sum, .processed = processed};
}
[[nodiscard]] inline DotAccumulator DotI16I16To64(const soa::I16View lhs, const soa::I16View rhs) noexcept {
  if (lhs.size() != rhs.size()) return DotAccumulator{.sum = simd::SplatI64(0), .processed = 0u, .size_match = false};
  if (lhs.empty()) return DotAccumulator{.sum = simd::SplatI64(0), .processed = 0u, .empty_input = true};
  simd::I64x sum = simd::SplatI64(0);
  u64 processed = 0u;
  std::size_t index = 0u;
  for (; index + simd::LaneCount <= lhs.size(); index += simd::LaneCount) {
    sum = AddWrap(sum, MulLow(__builtin_convertvector(detail::LoadI16x2(lhs.data() + index), simd::I64x),
                              __builtin_convertvector(detail::LoadI16x2(rhs.data() + index), simd::I64x)));
    processed += static_cast<u64>(simd::LaneCount);
  }
  if (index < lhs.size()) {
    sum = AddWrap(sum, MulLow(soa::detail::LoadTailI16AsI64(lhs, index), soa::detail::LoadTailI16AsI64(rhs, index)));
    processed += ::rund::math64::detail::ScalarSatU32(lhs.size() - index);
  }
  return DotAccumulator{.sum = sum, .processed = processed};
}
}  // namespace rund::math64::nn
