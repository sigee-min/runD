#pragma once

#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

struct MetalRuntimeStats final {
  rund::RuntimeStats runtime{};
  std::uint64_t library_compile_count = 0u;
  std::uint64_t library_cache_hit_count = 0u;
};

struct MetalMemoryStats final {
  std::uint64_t current{};
  std::uint64_t peak{};
  std::uint64_t cumulative{};
  std::uint64_t reused{};
};

[[nodiscard]] MetalRuntimeStats
ReadMetalRuntimeStats(const rund::AccelDevice &pick);
[[nodiscard]] MetalMemoryStats
ReadMetalMemoryStats(const rund::AccelDevice &pick) noexcept;
void ResetMetalRuntimeStats(const rund::AccelDevice &pick);

} // namespace rund::node::accel::detail
