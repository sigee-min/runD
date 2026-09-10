#pragma once

#include <rund/compute/backend.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace rund::compute {

struct DevicePipelineMemoryLimit final {
  std::uint64_t bytes{std::numeric_limits<std::uint64_t>::max()};
};

// Aggregate Pipeline-admission accounting for one Device. This is deliberately
// separate from Device::memory(), which reports actual allocator/backend
// telemetry rather than the conservative committed preparation charge.
struct DevicePipelineMemoryReport final {
  Backend backend{Backend::Unavailable};
  std::uint64_t capacity_bytes{};
  std::uint64_t committed_bytes{};
  std::uint64_t preparing_bytes{};
  std::uint64_t available_bytes{};
  std::uint64_t peak_committed_bytes{};
  std::uint64_t peak_preparing_bytes{};
  std::uint64_t peak_used_bytes{};
  std::uint64_t admission_count{};
  std::uint64_t commit_count{};
  std::uint64_t release_count{};
  std::uint64_t rejection_count{};

  [[nodiscard]] constexpr bool available() const noexcept {
    return backend != Backend::Unavailable;
  }

  [[nodiscard]] constexpr bool limited() const noexcept {
    return available() &&
           capacity_bytes != std::numeric_limits<std::uint64_t>::max();
  }
};

static_assert(std::is_trivially_copyable_v<DevicePipelineMemoryLimit>);
static_assert(std::is_trivially_copyable_v<DevicePipelineMemoryReport>);

} // namespace rund::compute
