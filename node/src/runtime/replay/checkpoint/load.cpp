#include "local.hpp"

#include "../artifact/format.hpp"
#include "../exception.hpp"

#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace rund::replay {

Load<Checkpoint> Checkpoint::load(const std::span<const std::byte> artifact,
                                  const Limits limits) noexcept {
  if (artifact.size() > limits.max_bytes) {
    return Load<Checkpoint>{Code::CheckpointEncodedCapacityExceeded,
                            std::nullopt};
  }
  if (limits.max_entries < 1u) {
    return Load<Checkpoint>{Code::CheckpointEntryCapacityExceeded,
                            std::nullopt};
  }
  try {
    node::replay_detail::artifact::Reader in{artifact};
    if (!in.header(node::replay_detail::artifact::Kind::Checkpoint)) {
      return Load<Checkpoint>{Code::CheckpointBadHeader, std::nullopt};
    }
    std::uint64_t previous_segment_count = 0u;
    std::uint64_t previous_input_position = 0u;
    std::uint64_t previous_prefix_hash = 0u;
    std::uint64_t previous_transcript_prefix_hash = 0u;
    std::uint64_t segment_record_hash = 0u;
    std::uint64_t segment_input_count = 0u;
    std::uint64_t segment_input_hash = 0u;
    std::uint64_t segment_transcript_hash = 0u;
    std::uint64_t state_schema = 0u;
    std::uint64_t state_size = 0u;
    std::uint64_t state_hash = 0u;
    std::uint64_t checkpoint_hash = 0u;
    if (!in.varuint(previous_segment_count) ||
        !in.varuint(previous_input_position) ||
        !in.fixed64(previous_prefix_hash) ||
        !in.fixed64(previous_transcript_prefix_hash) ||
        !in.fixed64(segment_record_hash) || !in.varuint(segment_input_count) ||
        !in.fixed64(segment_input_hash) ||
        !in.fixed64(segment_transcript_hash) || !in.varuint(state_schema) ||
        !in.varuint(state_size) || !in.fixed64(state_hash) ||
        !in.fixed64(checkpoint_hash)) {
      return Load<Checkpoint>{Code::CheckpointBadField, std::nullopt};
    }
    if (state_schema == 0u) {
      return Load<Checkpoint>{Code::StateSchemaInvalid, std::nullopt};
    }
    if (state_size > limits.max_state_bytes ||
        state_size > static_cast<std::uint64_t>(
                         std::numeric_limits<std::size_t>::max()) ||
        state_size != in.remaining()) {
      return Load<Checkpoint>{Code::CheckpointStateCapacityExceeded,
                              std::nullopt};
    }
    std::span<const std::byte> state_view{};
    if (!in.take(static_cast<std::size_t>(state_size), state_view) ||
        !in.done()) {
      return Load<Checkpoint>{Code::CheckpointBadField, std::nullopt};
    }
    std::vector<std::byte> state(static_cast<std::size_t>(state_size));
    if (!state.empty()) {
      std::memcpy(state.data(), state_view.data(), state.size());
    }

    const std::byte *const state_allocation = state.data();
    const std::shared_ptr<const Checkpoint::Data> candidate =
        std::make_shared<const Checkpoint::Data>(
            previous_segment_count, previous_input_position,
            previous_prefix_hash, previous_transcript_prefix_hash,
            segment_record_hash, segment_input_count, segment_input_hash,
            segment_transcript_hash, state_schema,
            OwnedCheckpointState::Adopt(state_schema, std::move(state)));
    if (state_size != 0u && candidate->state.data() != state_allocation) {
      return Load<Checkpoint>{Code::CheckpointStateAdoptionInvalid,
                              std::nullopt};
    }
    if (candidate->state_schema != state_schema ||
        candidate->state_size != state_size) {
      return Load<Checkpoint>{Code::CheckpointStateSizeInvalid, std::nullopt};
    }
    if (candidate->state_hash != state_hash) {
      return Load<Checkpoint>{Code::CheckpointStateHashInvalid, std::nullopt};
    }
    if (!candidate->valid()) {
      return Load<Checkpoint>{Code::CheckpointChainInvalid, std::nullopt};
    }
    if (candidate->checkpoint_hash != checkpoint_hash) {
      return Load<Checkpoint>{Code::CheckpointHashInvalid, std::nullopt};
    }
    return Load<Checkpoint>{Code::Ok, Checkpoint{std::move(candidate)}};
  } catch (...) {
    return Load<Checkpoint>{
        node::replay_detail::CurrentExceptionCode({
            .bad_alloc = Code::CheckpointCapacityExceeded,
            .length_error = Code::CheckpointCapacityExceeded,
            .unexpected = Code::CheckpointLoadFailed,
        }),
        std::nullopt};
  }
}

} // namespace rund::replay
