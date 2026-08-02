#include "../source.hpp"

#include "../source/program.hpp"
#include <kernel/program/compute/lowering/metal/fixed.hpp>

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool EmitMetalNumericFixedLane32SourceImpl(Sink &sink) {
  sink += R"MSL(
#include <metal_stdlib>
using namespace metal;
)MSL";
  rund::kernel::compute_lowering_detail::AppendMetalFixedLane32Helpers(sink);
  sink += R"MSL(

struct NumericParams {
  ulong op;
  ulong layout;
  ulong rows;
  ulong cols;
  ulong inner;
  ulong batch_count;
  ulong rhs_cols;
  ulong value_count;
  ulong vector_count;
  uint mode;
  uint aux;
  uint max_iterations;
  uint words;
};

constant uint RundMatrixArithmetic [[function_constant(0)]];
constant uint RundFixedFraction [[function_constant(1)]];
constant uint RundFixedRounding [[function_constant(2)]];
constant uint RundFixedOverflow [[function_constant(3)]];

inline int clamp_i32(long v) {
  return int(max(long(-2147483647 - 1), min(long(2147483647), v)));
}
inline uint mag_i32(int value) {
  return value < 0 ? uint(-long(value)) : uint(value);
}

inline int add_q31(int a, int b) { return RundAddSat32(a, b); }
inline int sub_q31(int a, int b) { return RundSubSat32(a, b); }
inline int fixed_i32(bool negative, ulong magnitude, bool overflow,
                     constant NumericParams& p) {
  if (RundFixedOverflow == 2u) {
    uint bits = uint(magnitude);
    return as_type<int>(negative ? 0u - bits : bits);
  }
  if (overflow || magnitude > (negative ? 0x80000000ul : 0x7ffffffful)) {
    return negative ? int(0x80000000u) : 2147483647;
  }
  uint bits = uint(magnitude);
  return as_type<int>(negative ? 0u - bits : bits);
}
inline int dynamic_mul_q31(constant NumericParams& p, int a, int b) {
  bool negative = (a < 0) != (b < 0);
  ulong product = ulong(mag_i32(a)) * ulong(mag_i32(b));
  ulong magnitude = product >> RundFixedFraction;
  ulong mask = (1ul << RundFixedFraction) - 1ul;
  ulong remainder = product & mask;
  ulong midpoint = 1ul << (RundFixedFraction - 1u);
  bool nearest = remainder > midpoint ||
                 (remainder == midpoint && (magnitude & 1ul) != 0ul);
  if ((RundFixedRounding == 2u && negative && remainder != 0ul) ||
      (RundFixedRounding == 3u && !negative && remainder != 0ul) ||
      (RundFixedRounding == 4u && nearest)) { ++magnitude; }
  return fixed_i32(negative, magnitude, magnitude > 0xfffffffful, p);
}
inline int matrix_add_i32(int a, int b, constant NumericParams& p) {
  return RundMatrixArithmetic == 1u
             ? add_q31(a, b)
             : as_type<int>(as_type<uint>(a) + as_type<uint>(b));
}
inline int matrix_mul_i32(int a, int b, constant NumericParams& p) {
  return RundMatrixArithmetic == 1u
             ? dynamic_mul_q31(p, a, b)
             : as_type<int>(as_type<uint>(a) * as_type<uint>(b));
}
inline int dynamic_div_q31(constant NumericParams& p, int a, int b) {
  if (b == 0) { return 0; }
  bool negative = (a < 0) != (b < 0);
  ulong numerator = ulong(mag_i32(a)) << RundFixedFraction;
  ulong denominator = ulong(mag_i32(b));
  ulong magnitude = numerator / denominator;
  ulong remainder = numerator % denominator;
  bool nearest = remainder * 2ul > denominator ||
                 (remainder * 2ul == denominator &&
                  (magnitude & 1ul) != 0ul);
  if ((RundFixedRounding == 2u && negative && remainder != 0ul) ||
      (RundFixedRounding == 3u && !negative && remainder != 0ul) ||
      (RundFixedRounding == 4u && nearest)) { ++magnitude; }
  return fixed_i32(negative, magnitude, magnitude > 0xfffffffful, p);
}
inline int dynamic_sqrt_q31(constant NumericParams& p, int value) {
  if (value <= 0) { return 0; }
  ulong target = ulong(uint(value)) << RundFixedFraction;
  ulong root = 0ul;
  for (int bit = 31; bit >= 0; --bit) {
    ulong candidate = root | (1ul << ulong(bit));
    if (candidate * candidate <= target) { root = candidate; }
  }
  ulong remainder = target - root * root;
  if (remainder != 0ul) {
    ulong upper_distance = root * 2ul + 1ul - remainder;
    bool nearest = remainder > upper_distance ||
                   (remainder == upper_distance && (root & 1ul) != 0ul);
    if (RundFixedRounding == 3u ||
        (RundFixedRounding == 4u && nearest)) { ++root; }
  }
  return fixed_i32(false, root, false, p);
}
inline int dynamic_one_q31(constant NumericParams& p) {
  return RundFixedFraction == 31u ? 2147483647
                                  : int(1u << RundFixedFraction);
}
inline uint dynamic_epsilon_q31(constant NumericParams& p) {
  uint one = uint(dynamic_one_q31(p));
  return one > (1u << 20u) ? one >> 20u : 1u;
}
#define mul_q31(a, b) dynamic_mul_q31(p, (a), (b))
#define div_q31(a, b) dynamic_div_q31(p, (a), (b))
#define sqrt_q31(a) dynamic_sqrt_q31(p, (a))
#define one_q31() dynamic_one_q31(p)
inline ulong midx(ulong r, ulong c, ulong rows, ulong cols, ulong layout) {
  return layout == 1ul ? r * cols + c : c * rows + r;
}
inline void rund_numeric_sync() {
  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
}

#define RUND_CAT_INNER(lhs, rhs) lhs##rhs
#define RUND_CAT(lhs, rhs) RUND_CAT_INNER(lhs, rhs)
#define RUND_SUFFIX i32
#define RUND_KERNEL(name) RUND_CAT(name, RUND_SUFFIX)
#define RUND_SCALAR int
#define RUND_MAGNITUDE uint
#define RUND_MAG_ZERO 0u
#define RUND_ZERO 0
#define RUND_ADD(lhs, rhs) add_q31((lhs), (rhs))
#define RUND_SUB(lhs, rhs) sub_q31((lhs), (rhs))
#define RUND_MUL(lhs, rhs) mul_q31((lhs), (rhs))
#define RUND_DIV(lhs, rhs) div_q31((lhs), (rhs))
#define RUND_SQRT(value) sqrt_q31((value))
#define RUND_ONE() one_q31()
#define RUND_ABS(value) RundAbsMagnitude32((value))
#define RUND_EPSILON(params) dynamic_epsilon_q31((params))
#define RUND_CLAMP_ABS(value) \
  int(min(RundAbsMagnitude32((value)), 0x7fffffffu))
#define RUND_NEGATIVE_ONE int(0x80000000u)
#define RUND_FROM_ULONG(value) int(value)
#define RUND_INDEX(row, col, rows, cols, layout) \
  midx((row), (col), (rows), (cols), (layout))
#define RUND_MATRIX_ADD(lhs, rhs, params) \
  matrix_add_i32((lhs), (rhs), (params))
#define RUND_MATRIX_MUL(lhs, rhs, params) \
  matrix_mul_i32((lhs), (rhs), (params))
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

bool EmitMetalNumericFixedLane32Source(
    backend_source_recipe::CountSink &sink) noexcept {
  return EmitMetalNumericFixedLane32SourceImpl(sink);
}

bool EmitMetalNumericFixedLane32Source(
    backend_source_recipe::StringSink &sink) {
  return EmitMetalNumericFixedLane32SourceImpl(sink);
}

} // namespace rund::node::accel::detail
