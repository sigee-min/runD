#pragma once

#include "../../../kernel/backend/source/sink.hpp"
#include "../../../scan/prefix.hpp"

#include "../../simd/source.hpp"

#include <string_view>

namespace rund::node::accel::detail {
namespace {

template <typename Sink>
[[nodiscard]] bool AppendMetalScanBaseSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  return source.append(R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint kScanWidth = )MSL") &&
         source.decimal(kScanPrefixWorkgroupWidth) && source.append("u;\n") &&
         source.append(MetalWideSimdPrefixSource) && source.append(R"MSL(
inline void rund_scan_exclusive_uint_pair(
    threadgroup uint* first, threadgroup uint* second, uint tid,
    uint block_size, uint simd_width, thread uint& exclusive,
    thread uint& total) {
  const uint value = first[tid];
  const uint local = simd_prefix_exclusive_sum(value);
  const uint local_total = simd_sum(value);
  const uint group = tid / simd_width;
  if (tid % simd_width == 0u) { second[group] = local_total; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    uint accumulated = 0u;
    for (uint i = 0u; i < (block_size + simd_width - 1u) / simd_width; ++i) {
      const uint subtotal = second[i];
      second[i] = accumulated;
      accumulated += subtotal;
    }
    first[0] = accumulated;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  exclusive = second[group] + local;
  total = first[0];
}
inline ulong rund_scan_exclusive_ulong(
    threadgroup ulong* groups, ulong value, uint tid, uint width,
    uint simd_width, uint physical_lane, thread ulong& exclusive) {
  const uint group = tid / simd_width;
  const uint last = simd_max(physical_lane);
  const ulong inclusive = rund_simd_prefix_u64(value);
  const ulong subtotal = rund_simd_last_u64(inclusive, last);
  if (tid % simd_width == 0u) { groups[group] = subtotal; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    ulong accumulated = 0ul;
    for (uint i = 0u; i < (width + simd_width - 1u) / simd_width; ++i) {
      const ulong next = groups[i];
      groups[i] = accumulated;
      accumulated += next;
    }
    groups[width] = accumulated;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  exclusive = groups[group] + inclusive - value;
  return groups[width];
}
inline bool rund_scan_overflow_uint(uint previous, uint value, uint next,
                                    uint signed_domain) {
  if (signed_domain == 0u) { return next < previous; }
  const bool same_sign = ((previous ^ value) & 0x80000000u) == 0u;
  const bool sign_changed = ((previous ^ next) & 0x80000000u) != 0u;
  return same_sign && sign_changed;
}
inline bool rund_scan_overflow_ulong(ulong previous, ulong value, ulong next,
                                     uint signed_domain) {
  if (signed_domain == 0u) { return next < previous; }
  const bool same_sign =
      ((previous ^ value) & 0x8000000000000000ul) == 0ul;
  const bool sign_changed =
      ((previous ^ next) & 0x8000000000000000ul) != 0ul;
  return same_sign && sign_changed;
}
)MSL");
}

} // namespace
} // namespace rund::node::accel::detail
