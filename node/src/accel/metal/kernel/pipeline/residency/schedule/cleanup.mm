#include "internal.hpp"

#include <memory>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

namespace schedule_detail {

MetalResidencyScheduleOwner::~MetalResidencyScheduleOwner() {
  for (MetalResidencyScheduleCommand &entry : commands) {
    if (entry.allocator != nil && (entry.completed || !active)) {
      [entry.allocator reset];
    }
    entry.command = nil;
    entry.allocator = nil;
  }
  service = nil;
}

} // namespace schedule_detail

#endif

#endif

rund::AccelCheck
AbortMetalResidencySchedule(const std::shared_ptr<void> &prepared,
                            const BackendResidencyWindowAbort &abort) noexcept {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
  if (@available(macOS 26.0, iOS 26.0, *)) {
    auto *const sequence = static_cast<MetalSequence *>(prepared.get());
    if (!ValidMetalSequence(sequence) || sequence->adapter == nullptr ||
        abort.failure.ok || abort.failure.reason == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    const std::shared_ptr<schedule_detail::MetalResidencyScheduleOwner> owner =
        schedule_detail::owner_of(sequence->residency_schedule.lock());
    if (owner == nullptr) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    std::scoped_lock lock{sequence->adapter->residency_terminal_gate,
                          owner->gate};
    if (!owner->active || owner->final_sent ||
        abort.plan_identity != owner->request.plan_identity ||
        abort.token != owner->request.token ||
        abort.generation != owner->request.generation) {
      return rund::AccelCheck{false, "accel_kernel_pipeline_invalid"};
    }
    owner->aborting = true;
    owner->abort_failure = abort.failure;
    std::size_t opened = 0u;
    for (schedule_detail::MetalResidencyScheduleCommand &entry :
         owner->commands) {
      if (!entry.signaled &&
          (entry.epoch < 2u || owner->commands[entry.epoch - 2u].completed) &&
          schedule_detail::signal_locked(owner, entry, abort.failure)) {
        ++opened;
      }
    }
    return opened == 0u
               ? rund::AccelCheck{false, "accel_kernel_pipeline_invalid"}
               : rund::AccelCheck{true, "ok"};
  }
#else
  static_cast<void>(prepared);
  static_cast<void>(abort);
#endif
  return rund::AccelCheck{false, "accel_metal_command_unavailable"};
}

} // namespace rund::node::accel::detail
