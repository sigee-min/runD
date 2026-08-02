#include "source.hpp"
#include "../kernel/source_recipe.hpp"

#include <kernel/program/compute/lowering/vulkan/fixed.hpp>

namespace rund::node::accel::detail {

namespace {

template <typename Sink>
[[nodiscard]] bool EmitParamsSource(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return sink.append(R"GLSL(
#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t op;
  uint64_t storage_layout;
  uint64_t rows;
  uint64_t cols;
  uint64_t inner;
  uint64_t batch_count;
  uint64_t rhs_cols;
  uint64_t value_count;
  uint64_t vector_count;
  uint mode;
  uint aux;
  uint max_iterations;
  uint words;
} p;

layout(constant_id = 0) const uint RundMatrixArithmetic = 1u;
layout(constant_id = 1) const uint RundFixedFraction = 1u;
layout(constant_id = 2) const uint RundFixedRounding = 1u;
layout(constant_id = 3) const uint RundFixedOverflow = 1u;
)GLSL");
}

template <typename Sink>
[[nodiscard]] bool EmitNumericBaseSource32(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!EmitParamsSource(sink)) {
    return false;
  }
  VulkanSourceTextSink out{sink};
  rund::kernel::compute_lowering_detail::AppendVulkanFixedLane32Helpers(out);
  out += R"GLSL(

int clamp_i32(int64_t v) {
  if (v > int64_t(2147483647)) { return 2147483647; }
  if (v < int64_t(-2147483648)) { return -2147483648; }
  return int(v);
}
uint mag_i32(int value) {
  return value < 0 ? uint(-(value + 1)) + 1u : uint(value);
}
int add_q31(int a, int b) { return int(RundAddSat32(uint(a), uint(b))); }
int sub_q31(int a, int b) { return int(RundSubSat32(uint(a), uint(b))); }
int fixed_i32(bool negative, uint64_t magnitude, bool overflow) {
  if (RundFixedOverflow == 2u) {
    uint bits = uint(magnitude);
    return int(negative ? 0u - bits : bits);
  }
  if (overflow || magnitude > (negative ? 0x80000000ul : 0x7ffffffful)) {
    return negative ? int(0x80000000u) : 2147483647;
  }
  return int(negative ? 0u - uint(magnitude) : uint(magnitude));
}
int mul_q31(int a, int b) {
  bool negative = (a < 0) != (b < 0);
  uint64_t product = uint64_t(mag_i32(a)) * uint64_t(mag_i32(b));
  uint64_t magnitude = product >> RundFixedFraction;
  uint64_t mask = (uint64_t(1) << RundFixedFraction) - uint64_t(1);
  uint64_t remainder = product & mask;
  uint64_t midpoint = uint64_t(1) << (RundFixedFraction - 1u);
  bool nearest = remainder > midpoint ||
                 (remainder == midpoint &&
                  (magnitude & uint64_t(1)) != 0ul);
  if ((RundFixedRounding == 2u && negative && remainder != 0ul) ||
      (RundFixedRounding == 3u && !negative && remainder != 0ul) ||
      (RundFixedRounding == 4u && nearest)) { ++magnitude; }
  return fixed_i32(negative, magnitude, magnitude > 0xfffffffful);
}
int matrix_add_i32(int a, int b) {
  return RundMatrixArithmetic == 1u ? add_q31(a, b)
                                    : int(uint(a) + uint(b));
}
int matrix_mul_i32(int a, int b) {
  return RundMatrixArithmetic == 1u ? mul_q31(a, b)
                                    : int(uint(a) * uint(b));
}
int div_q31(int a, int b) {
  if (b == 0) { return 0; }
  bool negative = (a < 0) != (b < 0);
  uint64_t numerator = uint64_t(mag_i32(a)) << RundFixedFraction;
  uint64_t denominator = uint64_t(mag_i32(b));
  uint64_t magnitude = numerator / denominator;
  uint64_t remainder = numerator % denominator;
  bool nearest = remainder * 2ul > denominator ||
                 (remainder * 2ul == denominator &&
                  (magnitude & uint64_t(1)) != 0ul);
  if ((RundFixedRounding == 2u && negative && remainder != 0ul) ||
      (RundFixedRounding == 3u && !negative && remainder != 0ul) ||
      (RundFixedRounding == 4u && nearest)) { ++magnitude; }
  return fixed_i32(negative, magnitude, magnitude > 0xfffffffful);
}
int sqrt_q31(int value) {
  if (value <= 0) { return 0; }
  uint64_t target = uint64_t(uint(value)) << RundFixedFraction;
  uint64_t root = 0ul;
  for (int bit = 31; bit >= 0; --bit) {
    uint64_t candidate = root | (uint64_t(1) << uint64_t(bit));
    if (candidate * candidate <= target) { root = candidate; }
  }
  uint64_t remainder = target - root * root;
  if (remainder != 0ul) {
    uint64_t upper_distance = root * 2ul + 1ul - remainder;
    bool nearest = remainder > upper_distance ||
                   (remainder == upper_distance &&
                    (root & uint64_t(1)) != 0ul);
    if (RundFixedRounding == 3u ||
        (RundFixedRounding == 4u && nearest)) { ++root; }
  }
  return fixed_i32(false, root, false);
}
int one_q31() {
  return RundFixedFraction == 31u ? 2147483647
                                  : int(1u << RundFixedFraction);
}
uint epsilon_q31() {
  uint one = uint(one_q31());
  return one > (1u << 20u) ? one >> 20u : 1u;
}
uint64_t midx(uint64_t r, uint64_t c, uint64_t rows, uint64_t cols,
              uint64_t storage_layout) {
  return storage_layout == uint64_t(1) ? r * cols + c : c * rows + r;
}
uint ix(uint64_t value) { return uint(value); }
)GLSL";
  return out.ok();
}

template <typename Sink>
[[nodiscard]] bool EmitNumericBaseSource64(Sink &sink)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!EmitParamsSource(sink)) {
    return false;
  }
  VulkanSourceTextSink out{sink};
  rund::kernel::compute_lowering_detail::AppendVulkanFixedLane64Helpers(out);
  out += R"GLSL(
uint64_t as_u64(int64_t value) { return uint64_t(value); }
int64_t as_i64(uint64_t value) { return int64_t(value); }
int64_t add_q63(int64_t a, int64_t b) {
  return as_i64(RundAddSat64(as_u64(a), as_u64(b)));
}
int64_t sub_q63(int64_t a, int64_t b) {
  return as_i64(RundSubSat64(as_u64(a), as_u64(b)));
}
int64_t mul_q63(int64_t a, int64_t b) {
  bool negative = (a < 0l) != (b < 0l);
  RundU128 product = RundMulWide64(RundAbsMagnitude64(as_u64(a)),
                                   RundAbsMagnitude64(as_u64(b)));
  uint64_t magnitude = (product.hi << (64u - RundFixedFraction)) |
                       (product.lo >> RundFixedFraction);
  bool overflow = (product.hi >> RundFixedFraction) != 0ul;
  uint64_t remainder =
      product.lo & ((uint64_t(1) << RundFixedFraction) - uint64_t(1));
  uint64_t midpoint = uint64_t(1) << (RundFixedFraction - 1u);
  bool nearest = remainder > midpoint ||
                 (remainder == midpoint && (magnitude & uint64_t(1)) != 0ul);
  if ((RundFixedRounding == 2u && negative && remainder != 0ul) ||
      (RundFixedRounding == 3u && !negative && remainder != 0ul) ||
      (RundFixedRounding == 4u && nearest)) {
    ++magnitude;
    overflow = overflow || magnitude == 0ul;
  }
  if (RundFixedOverflow == 2u) {
    return as_i64(negative ? 0ul - magnitude : magnitude);
  }
  if (overflow) {
    return negative ? int64_t(0x8000000000000000ul)
                    : int64_t(0x7ffffffffffffffful);
  }
  return as_i64(RundClampSignedMagnitude64(negative, magnitude));
}
int64_t matrix_add_i64(int64_t a, int64_t b) {
  return RundMatrixArithmetic == 1u ? add_q63(a, b)
                                    : as_i64(as_u64(a) + as_u64(b));
}
int64_t matrix_mul_i64(int64_t a, int64_t b) {
  return RundMatrixArithmetic == 1u ? mul_q63(a, b)
                                    : as_i64(as_u64(a) * as_u64(b));
}
int64_t div_q63(int64_t a, int64_t b) {
  if (b == 0l) { return 0l; }
  uint64_t lhs_mag = RundAbsMagnitude64(as_u64(a));
  uint64_t rhs_mag = RundAbsMagnitude64(as_u64(b));
  RundU128 numerator = RundMakeU128(
      lhs_mag >> (64u - RundFixedFraction), lhs_mag << RundFixedFraction);
  uint64_t magnitude = RundUnsignedDivU128ByU64(numerator, rhs_mag);
  bool negative = (a < 0l) != (b < 0l);
  if (numerator.hi < rhs_mag) {
    RundU128 product = RundMulWide64(magnitude, rhs_mag);
    uint64_t remainder = RundSubU128(numerator, product).lo;
    uint64_t complement = rhs_mag - remainder;
    bool nearest = remainder > complement ||
                   (remainder == complement &&
                    (magnitude & uint64_t(1)) != 0ul);
    if ((RundFixedRounding == 2u && negative && remainder != 0ul) ||
        (RundFixedRounding == 3u && !negative && remainder != 0ul) ||
        (RundFixedRounding == 4u && nearest)) { ++magnitude; }
  }
  return as_i64(RundFixedOverflow == 2u
                    ? (negative ? 0ul - magnitude : magnitude)
                    : RundClampSignedMagnitude64(negative, magnitude));
}
int64_t sqrt_q63(int64_t value) {
  if (value <= 0l) { return 0l; }
  RundU128 target = RundMakeU128(
      as_u64(value) >> (64u - RundFixedFraction),
      as_u64(value) << RundFixedFraction);
  uint64_t root = RundUnsignedSqrtU128ToU64(target);
  RundU128 remainder = RundSubU128(target, RundMulWide64(root, root));
  if (remainder.hi != 0ul || remainder.lo != 0ul) {
    RundU128 upper = RundMakeU128(0ul, root * 2ul + 1ul);
    RundU128 upper_distance = RundSubU128(upper, remainder);
    bool above = RundGeU128(remainder, upper_distance) &&
                 (remainder.hi != upper_distance.hi ||
                  remainder.lo != upper_distance.lo);
    bool tie = remainder.hi == upper_distance.hi &&
               remainder.lo == upper_distance.lo;
    bool nearest = above || (tie && (root & uint64_t(1)) != 0ul);
    if (RundFixedRounding == 3u ||
        (RundFixedRounding == 4u && nearest)) { ++root; }
  }
  return as_i64(RundFixedOverflow == 2u ? root
                                 : min(root, 0x7ffffffffffffffful));
}
int64_t one_q63() {
  return RundFixedFraction == 63u
             ? int64_t(0x7ffffffffffffffful)
             : int64_t(uint64_t(1) << RundFixedFraction);
}
uint64_t epsilon_q63() {
  uint64_t one = as_u64(one_q63());
  return one > (uint64_t(1) << 20u) ? one >> 20u : uint64_t(1);
}
uint64_t midx64(uint64_t r, uint64_t c, uint64_t rows, uint64_t cols,
                uint64_t storage_layout) {
  return storage_layout == uint64_t(1) ? r * cols + c : c * rows + r;
}
uint ix64(uint64_t value) { return uint(value); }
)GLSL";
  return out.ok();
}

template <typename Sink>
[[nodiscard]] bool EmitNumericBase(Sink &sink, const bool wide)
    noexcept(noexcept(sink.append(std::string_view{}))) {
  return wide ? EmitNumericBaseSource64(sink)
              : EmitNumericBaseSource32(sink);
}

} // namespace

bool EmitNumericBaseSource(backend_source_recipe::CountSink &sink,
                           const bool wide) noexcept {
  return EmitNumericBase(sink, wide);
}

bool EmitNumericBaseSource(backend_source_recipe::StringSink &sink,
                           const bool wide) {
  return EmitNumericBase(sink, wide);
}

bool NumericBaseSourceBytes(const bool wide, std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [wide](backend_source_recipe::CountSink &sink) noexcept {
        return EmitNumericBaseSource(sink, wide);
      },
      bytes);
}

std::string NumericBaseSource() {
  return backend_source_recipe::materialize(
      [](auto &sink) { return EmitNumericBaseSource(sink, false); });
}

std::string NumericBaseSource64() {
  return backend_source_recipe::materialize(
      [](auto &sink) { return EmitNumericBaseSource(sink, true); });
}

} // namespace rund::node::accel::detail
