#include "../../kernel/backend/source/storage.hpp"
#include "local.hpp"

#include <string_view>

namespace rund::node::accel::detail {

namespace {
inline constexpr std::string_view Source = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint kHistogramStatusOk = 0xffffffffu;
constant uint kHistogramReasonBinInvalid = 0u;

struct HistogramParams {
  ulong element_count;
  ulong bin_count;
};

kernel void rund_compute_histogram_clear(
    device uint* counts [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]],
    constant HistogramParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong i = ulong(gid);
  if (i < params.bin_count) {
    counts[i] = 0u;
  }
  if (gid == 0u) {
    atomic_store_explicit(&status[0], kHistogramStatusOk,
                          memory_order_relaxed);
  }
}

kernel void rund_compute_histogram_count(
    device const uint* bins [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant HistogramParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint width [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]]) {
#if RUND_HISTOGRAM_LOCAL
  {
    threadgroup atomic_uint local_counts[RUND_HISTOGRAM_LOCAL_BINS];
    for (uint bin = tid; ulong(bin) < params.bin_count; bin += width) {
      atomic_store_explicit(&local_counts[bin], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ulong i = ulong(gid); i < params.element_count; i += ulong(grid_size)) {
      const uint bin = bins[i];
      if (ulong(bin) < params.bin_count) {
        atomic_fetch_add_explicit(&local_counts[bin], 1u, memory_order_relaxed);
      } else {
        atomic_store_explicit(&status[0], kHistogramReasonBinInvalid,
                              memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint bin = tid; ulong(bin) < params.bin_count; bin += width) {
      const uint count = atomic_load_explicit(&local_counts[bin], memory_order_relaxed);
      if (count != 0u) {
        atomic_fetch_add_explicit(&counts[bin], count, memory_order_relaxed);
      }
    }
    return;
  }
#else
  const ulong i = ulong(gid);
  if (i >= params.element_count) { return; }
  const uint bin = bins[i];
  if (ulong(bin) >= params.bin_count) {
    atomic_store_explicit(&status[0], kHistogramReasonBinInvalid,
                          memory_order_relaxed);
    return;
  }
  // A uniform live SIMD cohort needs one device atomic, including partial
  // cohorts after a tail or invalid-bin rejection. Mixed cohorts stay direct.
  if (simd_all(bin == simd_broadcast_first(bin))) {
    const uint rank = simd_prefix_exclusive_sum(1u);
    const uint count = simd_sum(1u);
    if (rank == 0u) {
      atomic_fetch_add_explicit(&counts[bin], count, memory_order_relaxed);
    }
  } else {
    atomic_fetch_add_explicit(&counts[bin], 1u, memory_order_relaxed);
  }
#endif
}
)MSL";
} // namespace

namespace {
template <typename Sink>
bool EmitMetalHistogramSource(Sink &sink, const std::uint64_t bins) {
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  source += MetalHistogramLocal(bins) ? "#define RUND_HISTOGRAM_LOCAL 1\n"
                                      : "#define RUND_HISTOGRAM_LOCAL 0\n";
  source += "#define RUND_HISTOGRAM_LOCAL_BINS ";
  (void)source.decimal(kHistogramLocalBins);
  source += "\n";
  source += Source;
  return source.valid();
}
} // namespace

std::string MetalHistogramSource(const std::uint64_t bins) {
  return backend_source_recipe::materialize(
      [bins](auto &sink) { return EmitMetalHistogramSource(sink, bins); });
}

std::uint64_t
MetalHistogramSourceUpperBytes(const std::uint64_t bins) noexcept {
  std::uint64_t bytes = 0u;
  const auto emit = [bins](backend_source_recipe::CountSink &sink) noexcept {
    return EmitMetalHistogramSource(sink, bins);
  };
  return backend_source_recipe::bytes(emit, bytes) ? bytes : 0u;
}

} // namespace rund::node::accel::detail
