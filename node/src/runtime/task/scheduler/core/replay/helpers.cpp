#include "internal.hpp"

#include "../../hash.hpp"
#include "../../reactor/registry.hpp"

#include <kernel/core/checked.hpp>

#include <limits>
#include <utility>

namespace rund::node::scheduler_replay_detail {

MatchResult poison_input(SchedulerState &state,
                         const ::rund::replay::Code code) noexcept {
  const char *const reason = ::rund::replay::error(code).data();
  state.identity.host_replay_failed = true;
  state.identity.host_replay_reason = reason;
  state.identity.host_replay_payload_failed = true;
  state.identity.host_replay_payload_reason = reason;
  return MatchResult{.code = code};
}

MatchResult store_replay_input(SchedulerState &state,
                               const InputBinding &binding,
                               const InputSourceRange source_range,
                               PayloadBytes bytes,
                               const PayloadCapture payload) noexcept {
  if (binding.source == 0u) {
    return poison_input(state, ::rund::replay::Code::InputIdInvalid);
  }
  if (binding.schema == 0u) {
    return poison_input(state, ::rund::replay::Code::InputSchemaInvalid);
  }
  if (!payload) {
    return poison_input(state, ::rund::replay::Code::InputRecordFailed);
  }
  const std::uint64_t byte_count = static_cast<std::uint64_t>(bytes.size());
  const std::uint64_t retained =
      state.evidence.host_payload_store.logical_bytes();
  const std::uint64_t capacity =
      state.resources.limits.host_payload_capacity_bytes;
  if (retained > capacity || byte_count > capacity - retained) {
    return poison_input(state, ::rund::replay::Code::InputCapacityExceeded);
  }
  try {
    if (!state.evidence.host_payload_store.AppendInput(
            binding.source, binding.schema, binding.sequence, source_range,
            std::move(bytes), payload)) {
      return poison_input(state, ::rund::replay::Code::InputRecordFailed);
    }
  } catch (...) {
    return poison_input(state, ::rund::replay::Code::InputRecordFailed);
  }
  return MatchResult{.code = ::rund::replay::Code::Ok};
}

bool same_identity(const InputBinding &left,
                   const InputBinding &right) noexcept {
  return left.source == right.source && left.schema == right.schema;
}

bool to_size(const std::uint64_t value, std::size_t &out) noexcept {
  if (value >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  out = static_cast<std::size_t>(value);
  return true;
}

bool can_account_input(const SchedulerState &state,
                       const std::size_t byte_count) noexcept {
  static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
  const std::uint64_t bytes = static_cast<std::uint64_t>(byte_count);
  return rund::kernel::checked::add(state.evidence.input_consumed_bytes, bytes);
}

void account_input(SchedulerState &state,
                   const std::size_t byte_count) noexcept {
  state.evidence.input_consumed_bytes += static_cast<std::uint64_t>(byte_count);
  ++state.evidence.input_count;
}

std::uint64_t simulation_fingerprint(const SchedulerState &state) noexcept {
  std::uint64_t hash = kFnvOffset;
  MixHash(hash, state.identity.next_task_id);
  MixHash(hash, state.identity.next_scope_id);
  MixHash(hash, state.identity.next_spawn_index);
  MixHash(hash, state.identity.next_wait_id);
  MixHash(hash, state.identity.next_reactor_many_group_id);
  MixHash(hash,
          static_cast<std::uint64_t>(state.reactor.reactor_ready_sets.size()) +
              1u);
  MixHash(hash, state.identity.next_stop_source_id);
  MixHash(hash, state.identity.next_timer_sequence);
  MixHash(hash, state.identity.next_channel_id);
  MixHash(hash, state.resources.live_tasks.load(std::memory_order_acquire));
  MixHash(hash, state.resources.live_channels);
  MixHash(hash, state.resources.live_channel_buffer_slots);
  MixHash(hash, state.resources.live_channel_waits);
  MixHash(hash, state.ready.records.size());
  MixHash(hash, state.ready.timers.size());
  MixHash(hash, state.ready.join_waits.size());
  MixHash(hash, ReactorRegistrySize(state.reactor.reactor));
  return hash;
}

} // namespace rund::node::scheduler_replay_detail
