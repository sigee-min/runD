#include "arena.hpp"

#include "../device/state.hpp"

#include <algorithm>
#include <limits>

namespace rund::compute::detail::memory {

std::uint64_t arena_bytes(const DeviceState &device) noexcept {
  std::uint64_t bytes =
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  if (const AccelDeviceState *const accel = accel_device(device)) {
    bytes = accel->pick.backend_info.storage_bytes;
  }
  bytes = std::min(
      bytes,
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()));
  return bytes & ~(Alignment - 1u);
}

} // namespace rund::compute::detail::memory
