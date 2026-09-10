#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../state.hpp"

#include <algorithm>
#include <cstdio>
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
  stats.runtime.run.work.command_inflight_peak =
      std::max(stats.runtime.run.work.command_inflight_peak,
               adapter->residency_command_peak.load(std::memory_order_acquire));
  stats.runtime.outcome = rund::AccelOutcome{.ok = true, .reason = "ok"};
  return stats;
}

void ResetMetalRuntimeStats(const rund::AccelDevice &pick) {
  MetalAdapter *const adapter = AdapterFromPick(pick);
  if (adapter == nullptr) {
    return;
  }
  std::unique_lock terminal_lock{adapter->residency_terminal_gate};
  std::unique_lock<std::mutex> lock{adapter->mutex};
  adapter->host_readback_cv.wait(
      lock, [adapter] { return adapter->active_host_readbacks == 0u; });
  adapter->stats = MetalRuntimeStats{
      .runtime = rund::RuntimeStats{.outcome = {.ok = true, .reason = "ok"}}};
  adapter->residency_command_peak.store(
      adapter->residency_command_active.load(std::memory_order_acquire),
      std::memory_order_release);
}

void SetMetalLastError(MetalAdapter &adapter,
                       const char *const reason) noexcept {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  adapter.last_error = reason == nullptr ? "compute_backend_failed" : reason;
}

void SetMetalLastErrorDetail(MetalAdapter &adapter,
                             const char *const reason) noexcept {
  std::lock_guard<std::mutex> lock{adapter.mutex};
  const char *const source =
      reason == nullptr ? "compute_backend_failed" : reason;
  std::snprintf(adapter.last_error_detail.data(),
                adapter.last_error_detail.size(), "%s", source);
  adapter.last_error = adapter.last_error_detail.data();
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
