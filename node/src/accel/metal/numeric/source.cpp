#include "source.hpp"

#include "source/program.hpp"
#include <kernel/program/compute/lowering/metal/fixed.hpp>

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitMetalNumericFixedLane64SourceImpl(Sink &sink) {
  rund::kernel::compute_lowering_detail::AppendMetalFixedLane64Helpers(sink);
  sink += R"MSL(

inline long add_q63(long a, long b) { return RundAddSat64(a, b); }
inline long sub_q63(long a, long b) { return RundSubSat64(a, b); }
inline long dynamic_mul_q63(constant NumericParams& p, long a, long b) {
  bool negative = (a < 0l) != (b < 0l);
  RundU128 product = RundMulWide64(RundAbsMagnitude64(a),
                                   RundAbsMagnitude64(b));
  ulong magnitude = (product.hi << (64u - RundFixedFraction)) |
                    (product.lo >> RundFixedFraction);
  bool overflow = (product.hi >> RundFixedFraction) != 0ul;
  ulong remainder = product.lo & ((1ul << RundFixedFraction) - 1ul);
  ulong midpoint = 1ul << (RundFixedFraction - 1u);
  bool nearest = remainder > midpoint ||
                 (remainder == midpoint && (magnitude & 1ul) != 0ul);
  if ((RundFixedRounding == 2u && negative && remainder != 0ul) ||
      (RundFixedRounding == 3u && !negative && remainder != 0ul) ||
      (RundFixedRounding == 4u && nearest)) {
    ++magnitude;
    overflow = overflow || magnitude == 0ul;
  }
  if (RundFixedOverflow == 2u) {
    return RundAsSigned64(negative ? 0ul - magnitude : magnitude);
  }
  if (overflow) {
    return RundAsSigned64(negative ? 0x8000000000000000ul
                                   : 0x7ffffffffffffffful);
  }
  return RundClampSignedMagnitude64(negative, magnitude);
}
inline long matrix_add_i64(long a, long b, constant NumericParams& p) {
  return RundMatrixArithmetic == 1u
             ? add_q63(a, b)
             : as_type<long>(as_type<ulong>(a) + as_type<ulong>(b));
}
inline long matrix_mul_i64(long a, long b, constant NumericParams& p) {
  return RundMatrixArithmetic == 1u
             ? dynamic_mul_q63(p, a, b)
             : as_type<long>(as_type<ulong>(a) * as_type<ulong>(b));
}
inline long dynamic_div_q63(constant NumericParams& p, long a, long b) {
  if (b == 0l) { return 0l; }
  ulong lhs = RundAbsMagnitude64(a);
  ulong rhs = RundAbsMagnitude64(b);
  RundU128 numerator = RundMakeU128(lhs >> (64u - RundFixedFraction),
                                    lhs << RundFixedFraction);
  ulong magnitude = RundUnsignedDivU128ByU64(numerator, rhs);
  bool negative = (a < 0l) != (b < 0l);
  if (numerator.hi < rhs) {
    RundU128 product = RundMulWide64(magnitude, rhs);
    ulong remainder = RundSubU128(numerator, product).lo;
    ulong complement = rhs - remainder;
    bool nearest = remainder > complement ||
                   (remainder == complement && (magnitude & 1ul) != 0ul);
    if ((RundFixedRounding == 2u && negative && remainder != 0ul) ||
        (RundFixedRounding == 3u && !negative && remainder != 0ul) ||
        (RundFixedRounding == 4u && nearest)) { ++magnitude; }
  }
  return RundFixedOverflow == 2u
             ? RundAsSigned64(negative ? 0ul - magnitude : magnitude)
             : RundClampSignedMagnitude64(negative, magnitude);
}
inline long dynamic_sqrt_q63(constant NumericParams& p, long value) {
  if (value <= 0l) { return 0l; }
  ulong bits = RundAsUnsigned64(value);
  RundU128 target = RundMakeU128(bits >> (64u - RundFixedFraction),
                                 bits << RundFixedFraction);
  ulong root = RundUnsignedSqrtU128ToU64(target);
  RundU128 remainder = RundSubU128(target, RundMulWide64(root, root));
  if (remainder.hi != 0ul || remainder.lo != 0ul) {
    RundU128 upper = RundMakeU128(0ul, root * 2ul + 1ul);
    RundU128 upper_distance = RundSubU128(upper, remainder);
    bool above = RundGeU128(remainder, upper_distance) &&
                 (remainder.hi != upper_distance.hi ||
                  remainder.lo != upper_distance.lo);
    bool tie = remainder.hi == upper_distance.hi &&
               remainder.lo == upper_distance.lo;
    bool nearest = above || (tie && (root & 1ul) != 0ul);
    if (RundFixedRounding == 3u ||
        (RundFixedRounding == 4u && nearest)) { ++root; }
  }
  return RundFixedOverflow == 2u
             ? RundAsSigned64(root)
             : RundAsSigned64(min(root, 0x7ffffffffffffffful));
}
inline long dynamic_one_q63(constant NumericParams& p) {
  return RundFixedFraction == 63u
             ? RundAsSigned64(0x7ffffffffffffffful)
             : long(1ul << RundFixedFraction);
}
inline ulong dynamic_epsilon_q63(constant NumericParams& p) {
  ulong one = RundAsUnsigned64(dynamic_one_q63(p));
  return one > (1ul << 20u) ? one >> 20u : 1ul;
}
#define mul_q63(a, b) dynamic_mul_q63(p, (a), (b))
#define div_q63(a, b) dynamic_div_q63(p, (a), (b))
#define sqrt_q63(a) dynamic_sqrt_q63(p, (a))
#define one_q63() dynamic_one_q63(p)
inline ulong midx64(ulong r, ulong c, ulong rows, ulong cols, ulong layout) {
  return layout == 1ul ? r * cols + c : c * rows + r;
}
#define RUND_CAT_INNER(lhs, rhs) lhs##rhs
#define RUND_CAT(lhs, rhs) RUND_CAT_INNER(lhs, rhs)
#define RUND_SUFFIX i64
#define RUND_KERNEL(name) RUND_CAT(name, RUND_SUFFIX)
#define RUND_SCALAR long
#define RUND_MAGNITUDE ulong
#define RUND_MAG_ZERO 0ul
#define RUND_ZERO 0l
#define RUND_ADD(lhs, rhs) add_q63((lhs), (rhs))
#define RUND_SUB(lhs, rhs) sub_q63((lhs), (rhs))
#define RUND_MUL(lhs, rhs) mul_q63((lhs), (rhs))
#define RUND_DIV(lhs, rhs) div_q63((lhs), (rhs))
#define RUND_SQRT(value) sqrt_q63((value))
#define RUND_ONE() one_q63()
#define RUND_ABS(value) RundAbsMagnitude64((value))
#define RUND_EPSILON(params) dynamic_epsilon_q63((params))
#define RUND_CLAMP_ABS(value) \
  long(min(RundAbsMagnitude64((value)), 0x7ffffffffffffffful))
#define RUND_NEGATIVE_ONE RundAsSigned64(0x8000000000000000ul)
#define RUND_FROM_ULONG(value) long(value)
#define RUND_INDEX(row, col, rows, cols, layout) \
  midx64((row), (col), (rows), (cols), (layout))
#define RUND_MATRIX_ADD(lhs, rhs, params) \
  matrix_add_i64((lhs), (rhs), (params))
#define RUND_MATRIX_MUL(lhs, rhs, params) \
  matrix_mul_i64((lhs), (rhs), (params))
)MSL";
  if (!EmitMetalNumericProgramSource(sink)) {
    return false;
  }
  sink += R"MSL(
#undef RUND_MATRIX_MUL
#undef RUND_MATRIX_ADD
#undef RUND_INDEX
#undef RUND_FROM_ULONG
#undef RUND_NEGATIVE_ONE
#undef RUND_CLAMP_ABS
#undef RUND_EPSILON
#undef RUND_ABS
#undef RUND_ONE
#undef RUND_SQRT
#undef RUND_DIV
#undef RUND_MUL
#undef RUND_SUB
#undef RUND_ADD
#undef RUND_ZERO
#undef RUND_MAG_ZERO
#undef RUND_MAGNITUDE
#undef RUND_SCALAR
#undef RUND_KERNEL
#undef RUND_SUFFIX
#undef RUND_CAT
#undef RUND_CAT_INNER
)MSL";
  return sink.valid();
}

} // namespace

bool EmitMetalNumericFixedLane64Source(
    backend_source_recipe::CountSink &sink) noexcept {
  return EmitMetalNumericFixedLane64SourceImpl(sink);
}

bool EmitMetalNumericFixedLane64Source(
    backend_source_recipe::StringSink &sink) {
  return EmitMetalNumericFixedLane64SourceImpl(sink);
}

namespace {

template <typename Sink>
[[nodiscard]] bool EmitMetalNumericSource(Sink &sink) noexcept(
    noexcept(EmitMetalNumericFixedLane32Source(sink))) {
  return EmitMetalNumericFixedLane32Source(sink) &&
         EmitMetalNumericFixedLane64Source(sink) && sink.valid();
}

} // namespace

std::string MetalNumericSource() {
  const auto emit = [](auto &sink) noexcept(
                        noexcept(EmitMetalNumericSource(sink))) {
    return EmitMetalNumericSource(sink);
  };
  return backend_source_recipe::materialize(emit);
}

bool MetalNumericSourceUpperBytes(std::uint64_t &upper) noexcept {
  const auto emit = [](backend_source_recipe::CountSink &sink) noexcept {
    return EmitMetalNumericSource(sink);
  };
  return backend_source_recipe::bytes(emit, upper);
}

} // namespace rund::node::accel::detail
