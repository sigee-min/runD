#include "local.hpp"

namespace rund::node::scheduler_cancel {

StopSourceRecord* FindStopSource(
    std::vector<StopSourceRecord>& sources,
    const ::rund::detail::task::StopIdentity identity) noexcept {
  for (StopSourceRecord& source : sources) {
    if (source.identity == identity) {
      return &source;
    }
  }
  return nullptr;
}

const StopSourceRecord* FindStopSource(
    const std::vector<StopSourceRecord>& sources,
    const ::rund::detail::task::StopIdentity identity) noexcept {
  for (const StopSourceRecord& source : sources) {
    if (source.identity == identity) {
      return &source;
    }
  }
  return nullptr;
}

}  // namespace rund::node::scheduler_cancel

namespace rund::node {

::rund::detail::task::StopIdentity Scheduler::CreateStopSource() noexcept {
  EnsureCurrentCommit();
  try {
    const std::uint64_t id = state_->identity.next_stop_source_id++;
    constexpr std::uint64_t kInitialGeneration = 1u;
    const ::rund::detail::task::StopIdentity identity{
        .scheduler_id = state_->identity.scheduler_id,
        .source_identity = {
            .source_id = id,
            .generation = kInitialGeneration,
            .epoch = state_->identity.stop_source_epoch,
        },
    };
    state_->reactor.stop_sources.push_back(StopSourceRecord{
        .identity = identity,
        .requested = false,
    });
    return identity;
  } catch (...) {
    return {};
  }
}

task::StopState Scheduler::StopRequestedUnsequenced(
    const ::rund::detail::task::StopIdentity identity) const noexcept {
  if (!identity) {
    return task::StopState::fail(ReasonCode::TaskInvalid);
  }
  const StopSourceRecord *const source = scheduler_cancel::FindStopSource(
      state_->reactor.stop_sources, identity);
  if (source == nullptr) {
    return task::StopState::fail(ReasonCode::TaskInvalid);
  }
  return task::StopState::success(source->requested);
}

} // namespace rund::node
