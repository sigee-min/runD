#include "data.hpp"

#include "../checkpoint.hpp"
#include <kernel/core/checked.hpp>

#include <node/runtime/replay/hash.hpp>

#include <limits>
#include <memory>

namespace rund::replay {

Checkpoint
Binding::checkpoint(const Record &completed_segment,
                    const std::span<const std::byte> state) const noexcept {
  try {
    if (const Code binding = code(); binding != Code::Ok) {
      return Checkpoint{binding};
    }
    if (!checkpointable_) {
      return Checkpoint{Code::StateSchemaInvalid};
    }
    if (const Code code = detail::ready_code(completed_segment);
        code != Code::Ok) {
      return Checkpoint{code};
    }
    if (!completed_segment.data_->record.ok()) {
      return Checkpoint{Code::CheckpointSegmentFailed};
    }
    if (completed_segment.data_->record.start_hash !=
        node::replay_detail::GenesisStartHash()) {
      return Checkpoint{Code::RecordStartInvalid};
    }
    const node::RuntimeReplayRecord &record = completed_segment.data_->record;
    return Checkpoint{std::make_shared<const Checkpoint::Data>(
        0u, 0u, 0u, 0u, record.replay_hash, record.input_count,
        record.input_hash, record.transcript_hash, schema_,
        OwnedCheckpointState::Copy(schema_, state))};
  } catch (...) {
    return Checkpoint{Code::AllocationFailed};
  }
}

Checkpoint
Binding::advance(const Checkpoint &previous, const Record &completed_segment,
                 const std::span<const std::byte> state) const noexcept {
  try {
    if (const Code binding = code(); binding != Code::Ok) {
      return Checkpoint{binding};
    }
    if (!checkpointable_) {
      return Checkpoint{Code::StateSchemaInvalid};
    }
    if (!previous.data_) {
      return Checkpoint{previous.code()};
    }
    if (!previous.data_->valid()) {
      return Checkpoint{previous.code()};
    }
    if (const Code code = detail::ready_code(completed_segment);
        code != Code::Ok) {
      return Checkpoint{code};
    }
    if (!completed_segment.data_->record.ok()) {
      return Checkpoint{Code::CheckpointSegmentFailed};
    }
    if (previous.schema() != schema_) {
      return Checkpoint{Code::StateSchemaInvalid};
    }
    if (completed_segment.data_->record.start_hash !=
        node::replay_detail::StartHash(previous.data_->checkpoint_hash)) {
      return Checkpoint{Code::RecordStartMismatch};
    }
    const node::RuntimeReplayRecord &record = completed_segment.data_->record;
    if (!rund::kernel::checked::add(previous.data_->segment_count, 1u) ||
        !rund::kernel::checked::add(previous.data_->input_position,
                                    record.input_count)) {
      return Checkpoint{Code::CheckpointPositionOverflow};
    }
    return Checkpoint{std::make_shared<const Checkpoint::Data>(
        previous.data_->segment_count, previous.data_->input_position,
        previous.data_->prefix_hash, previous.data_->transcript_prefix_hash,
        record.replay_hash, record.input_count, record.input_hash,
        record.transcript_hash, schema_,
        OwnedCheckpointState::Copy(schema_, state))};
  } catch (...) {
    return Checkpoint{Code::AllocationFailed};
  }
}

} // namespace rund::replay
