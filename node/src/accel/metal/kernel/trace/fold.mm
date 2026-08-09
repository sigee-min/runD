#include "../trace.hpp"

#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
rund::AccelCheck FoldMetalDispatchTrace(MetalDispatchTrace &trace,
                                        id<MTLDevice> const device,
                                        rund::RuntimeStats &stats) noexcept {
  if (!trace.available() || trace.failed ||
      trace.cursor != trace.sample_count || device == nil) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  MTLTimestamp cpu_end = 0u;
  MTLTimestamp gpu_end = 0u;
  [device sampleTimestamps:&cpu_end gpuTimestamp:&gpu_end];
  if (cpu_end <= trace.cpu_start || gpu_end <= trace.gpu_start) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  const auto *const values =
      static_cast<const MTLCounterResultTimestamp *>([trace.values contents]);
  if (values == nullptr) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  const std::uint64_t cpu_span = cpu_end - trace.cpu_start;
  const std::uint64_t gpu_span = gpu_end - trace.gpu_start;
  std::uint64_t elapsed = 0u;
  std::uint64_t samples = 0u;
  for (NSUInteger index = 0u; index < trace.sample_count; index += 2u) {
    const std::uint64_t begin = values[index].timestamp;
    const std::uint64_t end = values[index + 1u].timestamp;
    if (begin == 0u || end == 0u || begin == MTLCounterErrorValue ||
        end == MTLCounterErrorValue || begin < trace.gpu_start || end < begin ||
        end > gpu_end) {
      return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
    }
    const unsigned __int128 scaled =
        static_cast<unsigned __int128>(end - begin) * cpu_span;
    const unsigned __int128 nanoseconds = scaled / gpu_span;
    const std::uint64_t duration =
        nanoseconds > std::numeric_limits<std::uint64_t>::max()
            ? std::numeric_limits<std::uint64_t>::max()
            : static_cast<std::uint64_t>(nanoseconds);
    ::rund::detail::counter::Accumulate(elapsed, duration);
    ::rund::detail::counter::Accumulate(samples, 1u);
  }
  stats.run.time.accel_kernel_ns = elapsed;
  stats.run.time.accel_timestamp_count = samples;
  stats.run.time.accel_timestamp_source =
      "metal_dispatch_counter_timestamp_calibrated";
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
