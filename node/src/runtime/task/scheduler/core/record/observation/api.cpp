#include "local.hpp"

namespace rund::node {

bool Scheduler::RecordHostEvent(::rund::host::Event event) noexcept {
  return CommitHostEvent(event).ok();
}

bool Scheduler::RecordHostEvents(
    const std::vector<::rund::host::Event> &events) noexcept {
  bool ok = true;
  for (const ::rund::host::Event &event : events) {
    ok = RecordHostEvent(event) && ok;
  }
  return ok;
}

bool Scheduler::RecordHostEventFromHostApi(::rund::host::Event event) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const bool recorded = RecordHostEvent(event);
  CompletePrimitiveCommit();
  return recorded;
}

bool Scheduler::RecordHostEventFromHostApi(
    ::rund::host::Event event,
    const replay_detail::payload::RawByteSource &source) noexcept {
  (void)TrapLaneOwnedSegmentPrimitive();
  EnsureCurrentCommit();
  const HostEventCommitResult committed = CommitHostEvent(event, &source);
  CompletePrimitiveCommit();
  return committed.ok();
}

} // namespace rund::node
