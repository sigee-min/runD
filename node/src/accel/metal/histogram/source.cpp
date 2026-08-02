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
    uint gid [[thread_position_in_grid]]) {
  const ulong i = ulong(gid);
  if (i >= params.element_count) { return; }
  const uint bin = bins[i];
  if (ulong(bin) >= params.bin_count) {
    atomic_store_explicit(&status[0], kHistogramReasonBinInvalid,
                          memory_order_relaxed);
    return;
  }
  atomic_fetch_add_explicit(&counts[bin], 1u, memory_order_relaxed);
}
)MSL";
} // namespace

std::string MetalHistogramSource() { return std::string{Source}; }

std::uint64_t MetalHistogramSourceUpperBytes() noexcept {
  return Source.size();
}

}  // namespace rund::node::accel::detail
