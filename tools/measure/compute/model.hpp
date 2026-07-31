#pragma once

#include <rund/compute.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace rund::measure::compute {

using Clock = std::chrono::steady_clock;
using ::rund::compute::Backend;
using ::rund::compute::Fixed;

inline constexpr std::array kBackends{Backend::Cpu, Backend::Metal,
                                      Backend::Vulkan};

[[nodiscard]] inline ::rund::compute::Target
TargetFor(const Backend backend) noexcept {
  switch (backend) {
  case Backend::Cpu:
    return ::rund::compute::Target::cpu();
  case Backend::Metal:
    return ::rund::compute::Target::metal();
  case Backend::Vulkan:
    return ::rund::compute::Target::vulkan();
  case Backend::Unavailable:
    std::abort();
  }
  std::abort();
}

[[nodiscard]] inline const char *Name(const Backend backend) noexcept {
  switch (backend) {
  case Backend::Cpu:
    return "cpu";
  case Backend::Metal:
    return "metal";
  case Backend::Vulkan:
    return "vulkan";
  case Backend::Unavailable:
    return "unavailable";
  }
  return "unknown";
}

[[nodiscard]] inline double Median(std::vector<double> &samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2u;
  return samples.size() % 2u == 0u
             ? samples[middle - 1u] +
                   (samples[middle] - samples[middle - 1u]) / 2.0
             : samples[middle];
}

struct WarmCounters final {
  std::uint64_t pipeline_compiles = 0u;
  std::uint64_t buffer_allocations = 0u;
  std::uint64_t descriptor_pool_creations = 0u;
  std::uint64_t descriptor_set_allocations = 0u;
  std::uint64_t uploaded_bytes = 0u;
  std::uint64_t download_events = 0u;
  std::uint64_t downloaded_bytes = 0u;
  std::uint64_t internal_roundtrip_bytes = 0u;
  std::uint64_t external_roundtrip_bytes = 0u;

  template <class Stats> constexpr void observe(const Stats &stats) noexcept {
    ::rund::detail::counter::Accumulate(pipeline_compiles,
                                        stats.pipeline_compiles);
    ::rund::detail::counter::Accumulate(buffer_allocations,
                                        stats.buffer_allocations);
    ::rund::detail::counter::Accumulate(download_events, stats.download_events);
    ::rund::detail::counter::Accumulate(uploaded_bytes, stats.uploaded_bytes);
    if constexpr (requires {
                    stats.descriptor_pool_creations;
                    stats.descriptor_set_allocations;
                    stats.downloaded_bytes;
                    stats.internal_roundtrip_bytes;
                    stats.external_roundtrip_bytes;
                  }) {
      ::rund::detail::counter::Accumulate(descriptor_pool_creations,
                                          stats.descriptor_pool_creations);
      ::rund::detail::counter::Accumulate(descriptor_set_allocations,
                                          stats.descriptor_set_allocations);
      ::rund::detail::counter::Accumulate(downloaded_bytes,
                                          stats.downloaded_bytes);
      ::rund::detail::counter::Accumulate(internal_roundtrip_bytes,
                                          stats.internal_roundtrip_bytes);
      ::rund::detail::counter::Accumulate(external_roundtrip_bytes,
                                          stats.external_roundtrip_bytes);
    }
  }

  [[nodiscard]] constexpr bool zero() const noexcept {
    return pipeline_compiles == 0u && buffer_allocations == 0u &&
           descriptor_pool_creations == 0u &&
           descriptor_set_allocations == 0u && uploaded_bytes == 0u &&
           download_events == 0u && downloaded_bytes == 0u;
  }
};

} // namespace rund::measure::compute
