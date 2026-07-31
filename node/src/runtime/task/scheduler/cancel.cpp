#include "cancel/local.hpp"

namespace rund::task {

stop_source stop_source::create() noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return {};
  }
  std::uint64_t source_id = 0u;
  std::uint64_t generation = 0u;
  std::uint64_t epoch = 0u;
  std::uint64_t scheduler_id = 0u;
  const Status result = scheduler->CreateStopSource(
      &scheduler_id, &source_id, &generation, &epoch);
  if (!result) {
    return {};
  }
  return stop_source{stop_token{scheduler_id, source_id, generation, epoch}};
}

Status stop_source::request_stop() const noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return Status::fail(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->RequestStop(token_.scheduler_id_, token_.source_id_,
                                token_.generation_, token_.epoch_);
}

StopState stop_token::state() const noexcept {
  node::Scheduler *const scheduler = node::Scheduler::Active();
  if (scheduler == nullptr) {
    return StopState::fail(ReasonCode::NodeRuntimeMissing);
  }
  return scheduler->StopRequested(scheduler_id_, source_id_, generation_,
                                  epoch_);
}

}  // namespace rund::task
