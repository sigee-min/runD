#pragma once

#include "core.hpp"

#include <cstdint>
#include <limits>

namespace rund::measure::compute {

struct WarmCounterSample final {
  std::uint64_t pipeline_compiles;
  std::uint64_t buffer_allocations;
  std::uint64_t descriptor_pool_creations;
  std::uint64_t descriptor_set_allocations;
  std::uint64_t download_events;
  std::uint64_t uploaded_bytes;
  std::uint64_t host_write_bytes;
  std::uint64_t downloaded_bytes;
  std::uint64_t internal_roundtrip_bytes;
  std::uint64_t external_roundtrip_bytes;
};

consteval bool WarmCounterContract() {
  WarmCounters counters{};
  if (!counters.zero()) {
    return false;
  }
  counters.observe(WarmCounterSample{1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u});
  counters.observe(WarmCounterSample{9u, 8u, 7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u});
  if (counters.pipeline_compiles != 10u || counters.buffer_allocations != 10u ||
      counters.descriptor_pool_creations != 10u ||
      counters.descriptor_set_allocations != 10u ||
      counters.download_events != 10u || counters.uploaded_bytes != 10u ||
      counters.host_write_bytes != 10u || counters.downloaded_bytes != 10u ||
      counters.internal_roundtrip_bytes != 10u ||
      counters.external_roundtrip_bytes != 10u || counters.zero()) {
    return false;
  }
  constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
  counters.observe(WarmCounterSample{maximum, maximum, maximum, maximum,
                                     maximum, maximum, maximum, maximum,
                                     maximum, maximum});
  return counters.pipeline_compiles == maximum &&
         counters.buffer_allocations == maximum &&
         counters.descriptor_pool_creations == maximum &&
         counters.descriptor_set_allocations == maximum &&
         counters.download_events == maximum &&
         counters.uploaded_bytes == maximum &&
         counters.host_write_bytes == maximum &&
         counters.downloaded_bytes == maximum &&
         counters.internal_roundtrip_bytes == maximum &&
         counters.external_roundtrip_bytes == maximum;
}

static_assert(WarmCounterContract());

} // namespace rund::measure::compute
