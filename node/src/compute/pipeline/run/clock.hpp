#pragma once

#include <chrono>
#include <cstdint>

namespace rund::compute::detail {

[[nodiscard]] inline std::uint64_t pipeline_clock() noexcept {
  const auto elapsed = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
}

} // namespace rund::compute::detail
