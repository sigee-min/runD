#pragma once

#include "../../../kernel/backend/source/sink.hpp"

namespace rund::node::accel::detail {

template <bool Wide, typename Sink>
[[nodiscard]] bool AppendMetalScanPrefixSource(Sink &sink) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  constexpr std::string_view type = Wide ? "ulong" : "uint";
  constexpr std::string_view zero = Wide ? "0ul" : "0u";
  constexpr std::string_view suffix = Wide ? "u64" : "u32";
  source += R"MSL(
kernel void rund_compute_scan_prefix_)MSL";
  source += suffix;
  source += R"MSL((
    device )MSL";
  source += type;
  source += R"MSL(* totals [[buffer(0)]],
    constant ulong& block_count [[buffer(1)]],
    uint physical_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint width [[threads_per_threadgroup]]) {
  const uint tid = simd_group * simd_width +
      simd_prefix_exclusive_sum(1u);
)MSL";
  if constexpr (Wide) {
    source += R"MSL(  threadgroup ulong chunks[kScanWidth + 1u];
)MSL";
  } else {
    source += R"MSL(  threadgroup uint chunks[2][kScanWidth];
)MSL";
  }
  source += R"MSL(  const ulong chunk_size =
      (block_count + ulong(width) - 1ul) / ulong(width);
  const ulong begin = min(ulong(tid) * chunk_size, block_count);
  const ulong end = min(begin + chunk_size, block_count);
  )MSL";
  source += type;
  source += R"MSL( running = )MSL";
  source += zero;
  source += R"MSL(;
  for (ulong block = begin; block < end; ++block) {
    const )MSL";
  source += type;
  source += R"MSL( total = totals[block];
    totals[block] = running;
    running += total;
  }
)MSL";
  if constexpr (!Wide) {
    source += R"MSL(  chunks[0][tid] = running;
)MSL";
  }
  source += R"MSL(  )MSL";
  source += type;
  source += R"MSL( offset = )MSL";
  source += zero;
  source += R"MSL(;
)MSL";
  if constexpr (Wide) {
    source += R"MSL(  rund_scan_exclusive_ulong(chunks, running, tid, width, simd_width,
                            physical_lane, offset);
)MSL";
  } else {
    source += R"MSL(  )MSL";
    source += type;
    source += R"MSL( total = )MSL";
    source += zero;
    source += R"MSL(;
  rund_scan_exclusive_uint_pair(chunks[0], chunks[1], tid, width, simd_width, offset,
                                total);
)MSL";
  }
  source += R"MSL(  for (ulong block = begin; block < end; ++block) {
    totals[block] += offset;
  }
}
)MSL";
  return source.valid();
}

} // namespace rund::node::accel::detail
