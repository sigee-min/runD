#include "../trace.hpp"

#include "../../runtime/timestamp.hpp"

#include <rund/counter.hpp>

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck FoldVulkanDispatchTrace(VulkanAdapter &adapter,
                                         VulkanDispatchTrace &trace,
                                         rund::RuntimeStats &stats) noexcept {
  if (!trace.ready || trace.queries == VK_NULL_HANDLE ||
      trace.query_count == 0u || trace.values.size() < trace.query_count) {
    return rund::AccelCheck{false, "compute_telemetry_trace_unavailable"};
  }
  const VkResult queried = vkGetQueryPoolResults(
      adapter.device, trace.queries, 0u, trace.query_count,
      static_cast<std::size_t>(trace.query_count) * sizeof(std::uint64_t),
      trace.values.data(), sizeof(std::uint64_t),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  if (queried != VK_SUCCESS) {
    return rund::AccelCheck{
        false,
        VulkanFailureReason(queried, "compute_telemetry_trace_unavailable")};
  }
  std::uint64_t elapsed = 0u;
  std::uint64_t samples = 0u;
  for (std::uint32_t index = 0u; index < trace.query_count; index += 2u) {
    const std::uint64_t ticks =
        VulkanTimestampTicks(trace.values[index], trace.values[index + 1u],
                             adapter.timestamp_valid_bits);
    const long double scaled =
        static_cast<long double>(ticks) *
        static_cast<long double>(adapter.timestamp_period_ns);
    const std::uint64_t duration =
        scaled >= static_cast<long double>(
                      std::numeric_limits<std::uint64_t>::max())
            ? std::numeric_limits<std::uint64_t>::max()
            : static_cast<std::uint64_t>(scaled);
    ::rund::detail::counter::Accumulate(elapsed, duration);
    ::rund::detail::counter::Accumulate(samples, 1u);
  }
  stats.run.time.accel_kernel_ns = elapsed;
  stats.run.time.accel_timestamp_count = samples;
  stats.run.time.accel_timestamp_source = "vulkan_dispatch_timestamp_query";
  ::rund::detail::counter::Accumulate(adapter.accel_kernel_ns, elapsed);
  ::rund::detail::counter::Accumulate(adapter.accel_timestamp_count, samples);
  adapter.accel_timestamp_source = "vulkan_dispatch_timestamp_query";
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
