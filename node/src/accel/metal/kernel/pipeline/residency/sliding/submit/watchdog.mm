#include "internal.hpp"

#include <atomic>
#include <limits>

namespace rund::node::accel::detail::metal_residency_sliding::submit {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK) &&                 \
    defined(__MAC_26_0) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) &&          \
    __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0

void ArmWatchdog(Context &context) noexcept {
  const std::uint64_t deadline_delta = context.native.force_terminal_loss
                                           ? FaultTerminalDeadlineNs
                                           : TerminalDeadlineNs;
  const std::uint64_t deadline =
      context.submit_begin >
              std::numeric_limits<std::uint64_t>::max() - deadline_delta
          ? std::numeric_limits<std::uint64_t>::max()
          : context.submit_begin + deadline_delta;
  context.native.watchdog_deadline_ns.store(deadline,
                                            std::memory_order_release);
  dispatch_source_set_timer(
      context.native.watchdog,
      dispatch_time(DISPATCH_TIME_NOW, static_cast<int64_t>(deadline_delta)),
      DISPATCH_TIME_FOREVER, 0u);
}

#endif

} // namespace rund::node::accel::detail::metal_residency_sliding::submit
