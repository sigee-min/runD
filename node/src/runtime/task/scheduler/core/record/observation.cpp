#include "observation/local.hpp"

namespace rund::node {

void Scheduler::RecordObservation(const task::ObservationKind kind,
                                  const ReasonCode code,
                                  const std::uint64_t task_id,
                                  const std::uint64_t wait_id, const int fd,
                                  const short interest, const short revents,
                                  const std::int64_t deadline_ns) noexcept {
  std::lock_guard lock{state_->evidence.mutex};
  state_->RequireSequencer();
  if (state_->plan.failure != ReasonCode::Ok) {
    return;
  }
  const std::uint64_t physical_sequence =
      state_->identity.next_observation_sequence++;
  task::Observation observation{
      .sequence = state_->plan.observation(physical_sequence),
      .kind = kind,
      .task_id = state_->plan.task(task_id),
      .wait_id = state_->plan.wait(wait_id),
      .fd = state_->plan.descriptor(fd),
      .interest = interest,
      .revents = revents,
      .deadline_ns = deadline_ns,
      .reason_code = code,
  };
  if (state_->plan.failure != ReasonCode::Ok) {
    return;
  }
  ::rund::detail::counter::Accumulate(
      ::rund::detail::task::Stat(state_->evidence.metrics,
                                 ::rund::detail::task::StatSlot::Observations),
      1u);
  if (state_->resources.limits.observation_capacity == 0u ||
      state_->evidence.observations.size() <
          state_->resources.limits.observation_capacity) {
    state_->evidence.observations.push_back(observation);
  } else {
    ::rund::detail::counter::Accumulate(
        ::rund::detail::task::Stat(
            state_->evidence.metrics,
            ::rund::detail::task::StatSlot::ObservationDropped),
        1u);
  }
  HashObservation(state_->evidence.metrics, observation);
}

} // namespace rund::node
