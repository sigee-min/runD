#include "model.hpp"

#include "../../../domain.hpp"
#include "../../../kernel/backend/source_recipe.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] const char *OpName(const rund::kernel::ReduceOp op) noexcept {
  switch (op) {
  case rund::kernel::ReduceOp::Sum:
    return "sum";
  case rund::kernel::ReduceOp::CountNonzero:
    return "count";
  case rund::kernel::ReduceOp::Min:
    return "min";
  case rund::kernel::ReduceOp::Max:
    return "max";
  }
  return "invalid";
}

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

} // namespace

template <typename Sink>
[[nodiscard]] bool EmitMetalSegmentedReduceSource(
    Sink &source, const rund::kernel::ReduceOp op,
    const rund::kernel::ComputeDomain domain) noexcept(
    noexcept(source += std::string_view{})) {
  const bool signed_domain = IsSignedDomain(domain);
  source += "#include <metal_stdlib>\n"
            "using namespace metal;\n";
  if (!AppendSegmentedReduceShaderModel(source)) {
    return false;
  }
  source += R"MSL(
struct RundSegmentedReduceParams {
  ulong count;
  ulong block_count;
  ulong segments_per_group;
};
kernel void rund_compute_segmented_reduce_classify(
    device const uint* heads [[buffer(0)]],
    device ulong* counts [[buffer(1)]],
    device atomic_uint* status [[buffer(4)]],
    constant RundSegmentedReduceParams& params [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[RUND_SEGMENT_INDEX_WIDTH];
  const ulong groups = min(params.block_count, ulong(RUND_SEGMENT_MAX_GROUPS));
  for (ulong block = ulong(group.x); block < params.block_count;
       block += groups) {
    const ulong index = block * ulong(RUND_SEGMENT_INDEX_WIDTH) + ulong(lane);
    const uint head = index < params.count ? heads[index] : 0u;
    if (index == 0ul && head != 1u) {
      atomic_fetch_or_explicit(&status[0], RUND_SEGMENT_INVALID,
                               memory_order_relaxed);
    }
    if (index < params.count && head > 1u) {
      atomic_fetch_or_explicit(&status[0], RUND_SEGMENT_INVALID,
                               memory_order_relaxed);
    }
    partial[lane] = head != 0u ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      if (lane < stride) { partial[lane] += partial[lane + stride]; }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) { counts[block] = ulong(partial[0]); }
  }
}
kernel void rund_compute_segmented_reduce_prefix(
    device const ulong* counts [[buffer(0)]],
    device ulong* offsets [[buffer(1)]],
    device ulong* segment_count [[buffer(2)]],
    device uint* dispatch [[buffer(3)]],
    constant RundSegmentedReduceParams& params [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]]) {
  threadgroup ulong partial[RUND_SEGMENT_INDEX_WIDTH];
  const ulong width = ulong(RUND_SEGMENT_INDEX_WIDTH);
  const ulong quotient = params.block_count / width;
  const ulong remainder = params.block_count % width;
  const ulong begin = quotient * ulong(lane) + min(ulong(lane), remainder);
  const ulong end = begin + quotient + (ulong(lane) < remainder ? 1ul : 0ul);
  ulong local = 0ul;
  for (ulong block = begin; block < end; ++block) {
    offsets[block] = local;
    local += counts[block];
  }
  partial[lane] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < RUND_SEGMENT_INDEX_WIDTH; stride <<= 1u) {
    const uint index = (lane + 1u) * (stride << 1u) - 1u;
    if (index < RUND_SEGMENT_INDEX_WIDTH) {
      partial[index] += partial[index - stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) { partial[RUND_SEGMENT_INDEX_WIDTH - 1u] = 0ul; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
       stride >>= 1u) {
    const uint index = (lane + 1u) * (stride << 1u) - 1u;
    if (index < RUND_SEGMENT_INDEX_WIDTH) {
      const ulong left = partial[index - stride];
      partial[index - stride] = partial[index];
      partial[index] += left;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong base = partial[lane];
  for (ulong block = begin; block < end; ++block) { offsets[block] += base; }
  threadgroup_barrier(mem_flags::mem_device);
  if (lane == 0u) {
    const ulong last = params.block_count - 1ul;
    const ulong segments = offsets[last] + counts[last];
    segment_count[0] = segments;
    const ulong groups =
        segments / params.segments_per_group +
        (segments % params.segments_per_group != 0ul ? 1ul : 0ul);
    dispatch[0] = uint(min(groups, ulong(RUND_SEGMENT_MAX_GROUPS)));
    dispatch[1] = 1u;
    dispatch[2] = 1u;
  }
}
kernel void rund_compute_segmented_reduce_scatter(
    device const uint* heads [[buffer(0)]],
    device const ulong* offsets [[buffer(1)]],
    device ulong* starts [[buffer(2)]],
    constant RundSegmentedReduceParams& params [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  threadgroup uint partial[RUND_SEGMENT_INDEX_WIDTH];
  const ulong groups = min(params.block_count, ulong(RUND_SEGMENT_MAX_GROUPS));
  for (ulong block = ulong(group.x); block < params.block_count;
       block += groups) {
    const ulong index = block * ulong(RUND_SEGMENT_INDEX_WIDTH) + ulong(lane);
    const uint head = index < params.count ? heads[index] : 0u;
    partial[lane] = head != 0u ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < RUND_SEGMENT_INDEX_WIDTH; stride <<= 1u) {
      const uint offset = (lane + 1u) * (stride << 1u) - 1u;
      if (offset < RUND_SEGMENT_INDEX_WIDTH) {
        partial[offset] += partial[offset - stride];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) { partial[RUND_SEGMENT_INDEX_WIDTH - 1u] = 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = RUND_SEGMENT_INDEX_WIDTH >> 1u; stride > 0u;
         stride >>= 1u) {
      const uint offset = (lane + 1u) * (stride << 1u) - 1u;
      if (offset < RUND_SEGMENT_INDEX_WIDTH) {
        const uint left = partial[offset - stride];
        partial[offset - stride] = partial[offset];
        partial[offset] += left;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (head != 0u) { starts[offsets[block] + ulong(partial[lane])] = index; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
)MSL";
  source += WideHelpers();
  AppendReduce(source, op, signed_domain ? "int" : "uint",
               signed_domain ? "i32" : "u32", false, signed_domain);
  AppendReduce(source, op, signed_domain ? "long" : "ulong",
               signed_domain ? "i64" : "u64", true, signed_domain);
  return source.valid();
}

std::string
MetalSegmentedReduceSource(const rund::kernel::ReduceOp op,
                           const rund::kernel::ComputeDomain domain) {
  const auto emit = [op, domain](auto &sink) noexcept(noexcept(
      EmitMetalSegmentedReduceSource(sink, op, domain))) {
    return EmitMetalSegmentedReduceSource(sink, op, domain);
  };
  return backend_source_recipe::materialize(emit);
}

bool MetalSegmentedReduceSourceUpperBytes(
    const rund::kernel::ReduceOp op,
    const rund::kernel::ComputeDomain domain,
    std::uint64_t &upper) noexcept {
  const auto emit = [op, domain](
                        backend_source_recipe::CountSink &sink) noexcept {
    return EmitMetalSegmentedReduceSource(sink, op, domain);
  };
  return backend_source_recipe::bytes(emit, upper);
}

std::string
MetalSegmentedReduceKey(const rund::kernel::SegmentedReducePlan &plan,
                        const rund::kernel::ComputeDomain domain) {
  std::string key = "segmented-reduce.";
  key += OpName(plan.op);
  key += IsSignedDomain(domain) ? ".i" : ".u";
  key += plan.element_bytes == sizeof(rund::kernel::u64) ? "64" : "32";
  return key;
}

std::string
MetalSegmentedReduceName(const rund::kernel::SegmentedReducePlan &plan,
                         const rund::kernel::ComputeDomain domain) {
  std::string name = "rund_compute_segmented_reduce_";
  name += IsSignedDomain(domain) ? "i" : "u";
  name += plan.element_bytes == sizeof(rund::kernel::u64) ? "64" : "32";
  return name;
}

#endif

} // namespace rund::node::accel::detail
