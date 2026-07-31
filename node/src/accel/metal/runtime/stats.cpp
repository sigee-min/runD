#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../state.hpp"
#include <mutex>

namespace rund::node::accel::detail {

namespace {

[[nodiscard]] MetalAdapter *
AdapterFromPick(const rund::AccelDevice &pick) noexcept {
  if (pick.api != rund::AccelApi::Metal || pick.backend.context == nullptr) {
    return nullptr;
  }
  return static_cast<MetalAdapter *>(pick.backend.context);
}

} // namespace

MetalRuntimeStats ReadMetalRuntimeStats(const rund::AccelDevice &pick) {
  MetalAdapter *const adapter = AdapterFromPick(pick);
  if (adapter == nullptr) {
    return {};
  }
  std::unique_lock<std::mutex> lock{adapter->mutex};
  adapter->host_readback_cv.wait(
      lock, [adapter] { return adapter->active_host_readbacks == 0u; });
  MetalRuntimeStats stats = adapter->stats;
  stats.ok = true;
  stats.reason = "ok";
  return stats;
}

void ResetMetalRuntimeStats(const rund::AccelDevice &pick) {
  MetalAdapter *const adapter = AdapterFromPick(pick);
  if (adapter == nullptr) {
    return;
  }
  std::unique_lock<std::mutex> lock{adapter->mutex};
  adapter->host_readback_cv.wait(
      lock, [adapter] { return adapter->active_host_readbacks == 0u; });
  adapter->stats = MetalRuntimeStats{.ok = true, .reason = "ok"};
}

void SetMetalLastError(MetalAdapter &adapter,
                       const char *const reason) noexcept {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  adapter.last_error = reason == nullptr ? "compute_backend_failed" : reason;
}

const char *MetalLastError(void *const context) noexcept {
  auto *const adapter = static_cast<MetalAdapter *>(context);
  if (adapter == nullptr) {
    return "compute_backend_failed";
  }
  std::lock_guard<std::mutex> lock{adapter->mutex};
  return adapter->last_error == nullptr ? "compute_backend_failed"
                                        : adapter->last_error;
}

} // namespace rund::node::accel::detail
