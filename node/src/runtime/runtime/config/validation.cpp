#include "../local.hpp"

#include <limits>

namespace rund::node::runtime_detail {

bool HostReplayStorageValid(const ::rund::replay::Storage &config) noexcept {
  if (config.max_bytes == 0u) {
    return false;
  }
  switch (config.mode) {
  case ::rund::replay::StorageMode::Memory:
    return config.directory.empty();
  case ::rund::replay::StorageMode::Spill:
    return config.segment_bytes != 0u && config.max_allocated_bytes != 0u &&
           !config.directory.empty() && config.budget;
  }
  return false;
}

bool HostReplayDiagnosticValid(
    const ::rund::replay::Diagnostic &config) noexcept {
  const bool bytes_enabled = config.window_bytes != 0u;
  const bool records_enabled = config.window_records != 0u;
  return bytes_enabled == records_enabled &&
         config.window_bytes <= static_cast<std::uint64_t>(
                                    std::numeric_limits<std::size_t>::max()) &&
         config.window_records <= static_cast<std::uint64_t>(
                                      std::numeric_limits<std::size_t>::max());
}

} // namespace rund::node::runtime_detail
