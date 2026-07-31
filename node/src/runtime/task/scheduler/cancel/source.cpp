#include "local.hpp"

namespace rund::node::scheduler_cancel {

StopSourceRecord* FindStopSource(
    std::vector<StopSourceRecord>& sources,
    const std::uint64_t scheduler_id,
    const std::uint64_t source_id,
    const std::uint64_t generation,
    const std::uint64_t epoch) noexcept {
  for (StopSourceRecord& source : sources) {
    if (source.scheduler_id == scheduler_id && source.id == source_id &&
        source.generation == generation && source.epoch == epoch) {
      return &source;
    }
  }
  return nullptr;
}

const StopSourceRecord* FindStopSource(
    const std::vector<StopSourceRecord>& sources,
    const std::uint64_t scheduler_id,
    const std::uint64_t source_id,
    const std::uint64_t generation,
    const std::uint64_t epoch) noexcept {
  for (const StopSourceRecord& source : sources) {
    if (source.scheduler_id == scheduler_id && source.id == source_id &&
        source.generation == generation && source.epoch == epoch) {
      return &source;
    }
  }
  return nullptr;
}

}  // namespace rund::node::scheduler_cancel

namespace rund::node {

task::Status Scheduler::CreateStopSource(std::uint64_t *const scheduler_id,
                                       std::uint64_t *const source_id,
                                       std::uint64_t *const generation,
                                       std::uint64_t *const epoch) noexcept {
  if (scheduler_id == nullptr || source_id == nullptr ||
      generation == nullptr || epoch == nullptr) {
    return task::Status::fail(ReasonCode::TaskInvalid);
  }
  EnsureCurrentCommit();
  try {
    const std::uint64_t id = state_->identity.next_stop_source_id++;
    constexpr std::uint64_t kInitialGeneration = 1u;
    state_->reactor.stop_sources.push_back(StopSourceRecord{
        .scheduler_id = state_->identity.scheduler_id,
        .id = id,
        .generation = kInitialGeneration,
        .epoch = state_->identity.stop_source_epoch,
        .requested = false,
    });
    *scheduler_id = state_->identity.scheduler_id;
    *source_id = id;
    *generation = kInitialGeneration;
    *epoch = state_->identity.stop_source_epoch;
  } catch (...) {
    *scheduler_id = 0u;
    *source_id = 0u;
    *generation = 0u;
    *epoch = 0u;
    return task::Status::fail(ReasonCode::TaskCapacityExceeded);
  }
  return task::Status::success();
}

task::StopState Scheduler::StopRequestedUnsequenced(
    const std::uint64_t scheduler_id, const std::uint64_t source_id,
    const std::uint64_t generation, const std::uint64_t epoch) const noexcept {
  if (scheduler_id == 0u || source_id == 0u || generation == 0u ||
      epoch == 0u) {
    return task::StopState::fail(ReasonCode::TaskInvalid);
  }
  const StopSourceRecord *const source = scheduler_cancel::FindStopSource(
      state_->reactor.stop_sources, scheduler_id, source_id, generation, epoch);
  if (source == nullptr) {
    return task::StopState::fail(ReasonCode::TaskInvalid);
  }
  return task::StopState::success(source->requested);
}

} // namespace rund::node
