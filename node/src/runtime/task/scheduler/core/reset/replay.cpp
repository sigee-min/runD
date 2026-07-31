#include "../../state/storage.hpp"

namespace rund::node {

void Scheduler::ClearReplay() noexcept {
  state_->evidence.input_capture_active.store(false, std::memory_order_release);
  state_->identity.next_expected_host_event = 0u;
  state_->identity.next_expected_host_payload = 0u;
  state_->identity.next_expected_replay_input = 0u;
  state_->identity.host_replay_failed = false;
  state_->identity.host_replay_reason = "ok";
  state_->identity.host_replay_payload_failed = false;
  state_->identity.host_replay_payload_reason = "ok";
  state_->evidence.observations.clear();
  state_->evidence.host_events.clear();
  state_->evidence.host_payload_store.Clear();
  state_->evidence.host_payload_reserved_bytes = 0u;
  state_->evidence.input_byte_size = 0u;
  state_->evidence.input_consumed_bytes = 0u;
  state_->evidence.input_count = 0u;
  state_->evidence.input_capture = {};
}

} // namespace rund::node
