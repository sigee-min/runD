#include "local.hpp"

namespace rund::node {

::rund::net::ready::many::Wait ReadyManyAccess::TryImmediate(
    Scheduler &scheduler, ReadyManyEntry &entry,
    const std::span<::rund::net::ready::Event> out,
    const std::optional<std::chrono::nanoseconds> timeout) noexcept {
  SchedulerState &state = *scheduler.state_;
  ReactorManyGroup immediate_group{
      .group_id = 0u,
      .task_id = entry.task_id,
      .first_request = 0u,
      .request_count = static_cast<std::uint32_t>(entry.requests.size()),
      .max_events = entry.output_limit,
  };
  const ReactorManyProbeResult immediate = ReactorProbeManyReady(
      scheduler, state.reactor.reactor.platform, entry.task_id,
      entry.output_limit, entry.requests,
      state.reactor.reactor_many_poll_request_scratch,
      state.reactor.reactor_many_ready_result_scratch, immediate_group,
      state.reactor.reactor_many_event_slots_scratch);
  std::uint32_t copied = 0u;
  if (immediate.total_ready() != 0u || !immediate.ok()) {
    ReasonCode immediate_code = immediate.code();
    const bool copied_ok = scheduler.CopyReactorManyEvents(0u, out, &copied);
    if (!copied_ok && immediate_code == ReasonCode::Ok) {
      immediate_code = ReasonCode::IoPollFailed;
    }
    ::rund::net::ready::many::Result result{immediate_code};
    result.events = copied;
    result.budget_exhausted = immediate.total_ready() > copied;
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
