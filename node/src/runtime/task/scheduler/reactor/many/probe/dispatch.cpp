#include "../local.hpp"

namespace rund::node {

[[nodiscard]] ::rund::net::ready::many::Wait FailManyCode(const ReasonCode code) noexcept {
  return ::rund::net::ready::many::detail::Access::Complete(
      ::rund::net::ready::many::Result{code});
}

[[nodiscard]] std::uint32_t
OutputLimit(const std::span<::rund::net::ready::Event> out,
            const ::rund::net::ready::many::Budget budget) noexcept {
  const std::size_t limit =
      std::min<std::size_t>(out.size(), budget.max_events);
  return limit > std::numeric_limits<std::uint32_t>::max()
             ? std::numeric_limits<std::uint32_t>::max()
             : static_cast<std::uint32_t>(limit);
}

bool Scheduler::CopyReactorManyEvents(const std::uint64_t group_id,
                                      const std::span<::rund::net::ready::Event> out,
                                      std::uint32_t *const copied) noexcept {
  if (copied == nullptr) {
    return false;
  }
  *copied = 0u;
  if (group_id == 0u) {
    const ReactorManyGroup immediate_group{
        .group_id = 0u,
        .first_request = 0u,
        .request_count = static_cast<std::uint32_t>(
            state_->reactor.reactor_many_event_slots_scratch.size()),
    };
    const bool copied_ok = ReactorManyEventSlotsCopy(
        immediate_group, state_->reactor.reactor_many_event_slots_scratch, out,
        copied);
    RecordReactorReadyManyEvents(state_->evidence.metrics, *copied);
    return copied_ok;
  } else if (const ReactorManyGroup *const group = ReactorManyFindGroup(
                 state_->reactor.reactor_many_groups, group_id);
             group != nullptr) {
    const bool copied_ok = ReactorManyEventSlotsCopy(
        *group, state_->reactor.reactor_many_event_slots, out, copied);
    RecordReactorReadyManyEvents(state_->evidence.metrics, *copied);
    return copied_ok;
  }
  RecordReactorReadyManyEvents(state_->evidence.metrics, *copied);
  return false;
}

} // namespace rund::node
