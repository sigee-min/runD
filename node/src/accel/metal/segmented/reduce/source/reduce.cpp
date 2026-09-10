#include "reduce.hpp"

#include "../../../../domain.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] const char *WideHelpers() noexcept {
  return R"MSL(
struct RundWide { ulong lo; ulong hi; };
inline RundWide rund_wide_add(RundWide lhs, RundWide rhs) {
  const ulong lo = lhs.lo + rhs.lo;
  return RundWide{lo, lhs.hi + rhs.hi + (lo < lhs.lo ? 1ul : 0ul)};
}
inline RundWide rund_wide_i32(int value) {
  return RundWide{ulong(long(value)), value < 0 ? ulong(-1) : 0ul};
}
inline RundWide rund_wide_u32(uint value) {
  return RundWide{ulong(value), 0ul};
}
inline RundWide rund_wide_i64(long value) {
  return RundWide{ulong(value), value < 0 ? ulong(-1) : 0ul};
}
inline RundWide rund_wide_u64(ulong value) {
  return RundWide{value, 0ul};
}
inline bool rund_wide_fits_i32(RundWide value) {
  return (value.hi == 0ul && value.lo <= 0x7ffffffful) ||
         (value.hi == ulong(-1) && value.lo >= 0xffffffff80000000ul);
}
inline bool rund_wide_fits_u32(RundWide value) {
  return value.hi == 0ul && value.lo <= 0xfffffffful;
}
inline bool rund_wide_fits_i64(RundWide value) {
  return (value.hi == 0ul && value.lo <= 0x7ffffffffffffffful) ||
         (value.hi == ulong(-1) && value.lo >= 0x8000000000000000ul);
}
inline bool rund_wide_fits_u64(RundWide value) {
  return value.hi == 0ul;
}
)MSL";
}

template <typename Sink>
void AppendReduce(Sink &source, const rund::kernel::ReduceOp op,
                  const char *const type, const char *const suffix,
                  const bool wide, const bool signed_domain) {
  source += "kernel void rund_compute_segmented_reduce_";
  source += suffix;
  source += "(device const ";
  source += type;
  source += R"MSL(* input [[buffer(0)]],
    device const ulong* starts [[buffer(1)]],
    device const ulong* segment_count [[buffer(2)]],
    device )MSL";
  source += type;
  source += R"MSL(* output [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant RundSegmentedReduceParams& params [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_count [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const ulong segments = segment_count[0];
  const ulong requested =
      segments / params.segments_per_group +
      (segments % params.segments_per_group != 0ul ? 1ul : 0ul);
  const ulong groups = min(requested, ulong(RUND_SEGMENT_MAX_GROUPS));
  if (groups == 0ul) { return; }
  const ulong stride = groups * ulong(simd_count);
)MSL";
  if (op == rund::kernel::ReduceOp::Sum ||
      op == rund::kernel::ReduceOp::CountNonzero) {
    source += R"MSL(  threadgroup RundWide partial[RUND_SEGMENT_INDEX_WIDTH];
  for (ulong slot = ulong(group.x) * ulong(simd_count) + ulong(simd);
       slot < segments;) {
    const ulong begin = starts[slot];
    const ulong end =
        slot + 1ul < segments ? starts[slot + 1ul] : params.count;
    RundWide acc = RundWide{0ul, 0ul};
    for (ulong index = begin + ulong(lane); index < end;
         index += ulong(simd_width)) {
)MSL";
    if (op == rund::kernel::ReduceOp::CountNonzero) {
      source += "      if (input[index] != 0) { acc = rund_wide_add(acc, "
                "RundWide{1ul, 0ul}); }\n";
    } else {
      source += "      acc = rund_wide_add(acc, rund_wide_";
      source += signed_domain ? "i" : "u";
      source += wide ? "64" : "32";
      source += "(input[index]));\n";
    }
    source += R"MSL(    }
    partial[tid] = acc;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = simd_width >> 1u; offset > 0u; offset >>= 1u) {
      if (lane < offset) {
        acc = rund_wide_add(acc, partial[tid + offset]);
      }
      partial[tid] = acc;
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
      const RundWide total = acc;
      if (!rund_wide_fits_)MSL";
    source += signed_domain ? "i" : "u";
    source += wide ? "64" : "32";
    source += "(total)) { atomic_fetch_or_explicit(&status[0], ";
    source += op == rund::kernel::ReduceOp::CountNonzero
                  ? "RUND_SEGMENT_COUNT_OVERFLOW"
                  : "RUND_SEGMENT_SUM_OVERFLOW";
    source += ", memory_order_relaxed); }\n      output[slot] = ";
    source += type;
    source += R"MSL((total.lo);
    }
    if (segments - slot <= stride) { break; }
    slot += stride;
  }
}
)MSL";
    return;
  }
  const char *const maximum =
      wide ? (signed_domain ? "0x7fffffffffffffffL" : "0xffffffffffffffffUL")
           : (signed_domain ? "0x7fffffff" : "0xffffffffU");
  const char *const minimum =
      wide ? (signed_domain ? "(-0x7fffffffffffffffL - 1L)" : "0ul")
           : (signed_domain ? "(-0x7fffffff - 1)" : "0u");
  source += "  threadgroup ";
  source += type;
  source += R"MSL( partial[RUND_SEGMENT_INDEX_WIDTH];
  for (ulong slot = ulong(group.x) * ulong(simd_count) + ulong(simd);
       slot < segments;) {
    const ulong begin = starts[slot];
    const ulong end =
        slot + 1ul < segments ? starts[slot + 1ul] : params.count;
    )MSL";
  source += type;
  source += " acc = ";
  source += op == rund::kernel::ReduceOp::Min ? maximum : minimum;
  source += R"MSL(;
    for (ulong index = begin + ulong(lane); index < end;
         index += ulong(simd_width)) {
      acc = )MSL";
  source += op == rund::kernel::ReduceOp::Min ? "min" : "max";
  source += R"MSL((acc, input[index]);
    }
    partial[tid] = acc;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = simd_width >> 1u; offset > 0u; offset >>= 1u) {
      if (lane < offset) {
        acc = )MSL";
  source += op == rund::kernel::ReduceOp::Min ? "min" : "max";
  source += R"MSL((acc, partial[tid + offset]);
      }
      partial[tid] = acc;
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) { output[slot] = acc; }
    if (segments - slot <= stride) { break; }
    slot += stride;
  }
}
)MSL";
}

template <typename Sink>
[[nodiscard]] bool EmitReduceSource(
    Sink &source, const rund::kernel::ReduceOp op,
    const rund::kernel::ComputeDomain domain) noexcept(
    noexcept(source += std::string_view{})) {
  const bool signed_domain = IsSignedDomain(domain);
  source += WideHelpers();
  AppendReduce(source, op, signed_domain ? "int" : "uint",
               signed_domain ? "i32" : "u32", false, signed_domain);
  AppendReduce(source, op, signed_domain ? "long" : "ulong",
               signed_domain ? "i64" : "u64", true, signed_domain);
  return source.valid();
}

} // namespace

bool EmitMetalSegmentedReduceReduceSource(
    backend_source_recipe::CountSink &sink, const rund::kernel::ReduceOp op,
    const rund::kernel::ComputeDomain domain) noexcept {
  return EmitReduceSource(sink, op, domain);
}

bool EmitMetalSegmentedReduceReduceSource(
    backend_source_recipe::StringSink &sink, const rund::kernel::ReduceOp op,
    const rund::kernel::ComputeDomain domain) {
  return EmitReduceSource(sink, op, domain);
}

#endif

} // namespace rund::node::accel::detail
