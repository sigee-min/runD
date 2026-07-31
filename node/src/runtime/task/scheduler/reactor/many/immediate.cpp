#include "local.hpp"

namespace rund::node {

::rund::net::ready::many::Wait ReadyManyAccess::TryImmediate(
    Scheduler &scheduler, ReadyManyEntry &entry,
    const std::span<::rund::net::ready::Event> out,
    const std::optional<std::chrono::nanoseconds> timeout) noexcept {
  SchedulerState &state = *scheduler.state_;
  std::uint32_t immediate_total = 0u;
  ReasonCode immediate_code = ReasonCode::Ok;
  const bool immediate_ok = scheduler.ProbeReactorManyReady(
      entry.requests, entry.task_id, 0u, entry.output_limit, &immediate_total,
      &immediate_code);
  std::uint32_t copied = 0u;
  if (immediate_total != 0u || !immediate_ok) {
    const bool copied_ok = scheduler.CopyReactorManyEvents(0u, out, &copied);
    if (!copied_ok && immediate_code == ReasonCode::Ok) {
      immediate_code = ReasonCode::IoPollFailed;
    }
    ::rund::net::ready::many::Result result{immediate_code};
    result.events = copied;
    result.budget_exhausted = immediate_total > copied;
    state.reactor.reactor_many_event_slots_scratch.clear();
    scheduler.CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }
  state.reactor.reactor_many_event_slots_scratch.clear();
  if (!timeout.has_value()) {
    const ::rund::net::ready::many::Result result{ReasonCode::Ok};
    scheduler.CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }
  if (timeout->count() == 0) {
    const ::rund::net::ready::many::Result result{ReasonCode::IoTimedOut};
    scheduler.CompletePrimitiveCommit();
    return ::rund::net::ready::many::detail::Access::Complete(result);
  }
  return ::rund::net::ready::many::detail::Access::Complete(
      ::rund::net::ready::many::Result{ReasonCode::NetReadyManyNotReady});
}

} // namespace rund::node
