#include "internal.hpp"

#include "../../../../../clock.hpp"

#include <cstring>
#include <utility>

namespace rund::node::accel::detail::vulkan_device_vsm {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

void complete_native(void *const raw, const KernelResult result) noexcept {
  auto *const owner = static_cast<Owner *>(raw);
  if (owner == nullptr) {
    return;
  }
  const std::uint64_t callback_ns = MonotonicNanoseconds();
  DeviceVsmRequest request{};
  DeviceVsmNativeExecution native{};
  VulkanAdapter *adapter = nullptr;
  std::uint64_t submit_begin_ns = 0u;
  {
    std::lock_guard lock{owner->gate};
    native = owner->pending_native;
    native.accepted = result.check;
    native.completed_ns = result.stats.run.time.command_submit_wait_ns;
    native.kernel_ns = result.stats.run.time.accel_kernel_ns;
    native.kernel_samples = result.stats.run.time.accel_timestamp_count;
    native.submit_wait_ns = result.stats.run.time.command_submit_wait_ns;
    native.native_check_ok = result.check.ok;
    native.native_check_code = result.check.first_status;
    native.native_check_reason = device_vsm_reason_code(result.check.reason);
    native.result_mapped = owner->result.buffer.mapped != nullptr;
    native.result_acquired = native.native_check_ok && native.result_mapped;
    if (native.result_acquired) {
      std::memcpy(native.counters.data(), owner->result.buffer.mapped,
                  sizeof(native.counters));
    }
    request = std::move(owner->pending_request);
    adapter = owner->adapter;
    submit_begin_ns = owner->submit_begin_ns;
    owner->submit_begin_ns = 0u;
    owner->in_flight = false;
  }
  DeviceVsmFinal final = ClassifyDeviceVsmTerminal(request, native);
  final.evidence.submit_ns = submit_begin_ns;
  final.evidence.callback_ns = callback_ns;
  if (final.terminal == DeviceVsmTerminal::UnknownMayWrite &&
      adapter != nullptr) {
    adapter->residency_quarantined.store(true, std::memory_order_release);
  }
  const std::uint64_t now = MonotonicNanoseconds();
  final.evidence.completed_ns =
      submit_begin_ns != 0u && now >= submit_begin_ns
          ? now - submit_begin_ns
          : 0u;
  if (request.final != nullptr) {
    request.final(request.user, std::move(final));
  }
}

} // namespace

rund::AccelCheck submit(const DeviceVsmRequest &request) noexcept {
  const std::shared_ptr<Owner> owner = owner_of(request.lowering);
  if (owner == nullptr || owner->magic != OwnerMagic ||
      owner->proof != request.proof ||
      !device_vsm_request_valid(owner->capability, request)) {
    return {false, "accel_kernel_pipeline_invalid"};
  }
  std::unique_lock owner_lock{owner->gate};
  if (owner->adapter == nullptr) {
    return {false, "compute_pipeline_busy"};
  }
  VulkanAdapter &adapter = *owner->adapter;
  std::unique_lock adapter_lock{adapter.mutex};
  if (adapter.residency_quarantined.load(std::memory_order_acquire)) {
    return {false, "compute_device_lost"};
  }
  if (owner->in_flight || owner->submitted || owner->pipeline == nullptr) {
    return {false, "compute_pipeline_busy"};
  }

  owner->submitted = true;
  owner->in_flight = true;
  owner->pending_request = request;
  const DeviceVsmNativeExecution native =
      execute(*owner, complete_native, owner.get());
  if (!native.accepted.ok) {
    owner->submitted = false;
    owner->pending_request = {};
    owner->submit_begin_ns = 0u;
    owner->in_flight = false;
    return native.accepted;
  }
  adapter_lock.unlock();
  owner_lock.unlock();
  return {true, "ok"};
}

#endif

} // namespace rund::node::accel::detail::vulkan_device_vsm
