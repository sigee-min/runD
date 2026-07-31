#include "checkpoint.hpp"

#include <node/runtime/replay/hash.hpp>

#include "artifact/format.hpp"
#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace rund::replay {
namespace {

[[nodiscard]] std::uint64_t BeginHash(const std::string_view domain) noexcept {
  std::uint64_t hash = node::replay_detail::kFnvOffset;
  node::replay_detail::MixString(hash, domain);
  return hash;
}

[[nodiscard]] std::uint64_t
HashCheckpointState(const std::uint64_t schema,
                    const std::span<const std::byte> state) noexcept {
  std::uint64_t hash = BeginHash("rund.replay.checkpoint.state");
  node::replay_detail::Mix(hash, schema);
  node::replay_detail::Mix(hash, static_cast<std::uint64_t>(state.size()));
  for (const std::byte value : state) {
    node::replay_detail::Mix(hash, std::to_integer<std::uint8_t>(value));
  }
  return hash;
}

[[nodiscard]] std::uint64_t HashCheckpointBoundary(
    const std::uint64_t segment_count, const std::uint64_t record_hash,
    const std::uint64_t input_count, const std::uint64_t input_hash,
    const std::uint64_t transcript_hash) noexcept {
  std::uint64_t hash = BeginHash("rund.replay.checkpoint.boundary");
  node::replay_detail::Mix(hash, segment_count);
  node::replay_detail::Mix(hash, record_hash);
  node::replay_detail::Mix(hash, input_count);
  node::replay_detail::Mix(hash, input_hash);
  node::replay_detail::Mix(hash, transcript_hash);
  return hash;
}

[[nodiscard]] std::uint64_t
HashCheckpointPrefix(const std::uint64_t previous_prefix_hash,
                     const std::uint64_t segment_count,
                     const std::uint64_t boundary_hash) noexcept {
  std::uint64_t hash = BeginHash("rund.replay.checkpoint.prefix");
  node::replay_detail::Mix(hash, previous_prefix_hash);
  node::replay_detail::Mix(hash, segment_count);
  node::replay_detail::Mix(hash, boundary_hash);
  return hash;
}

[[nodiscard]] std::uint64_t
HashCheckpointTranscript(const std::uint64_t previous_transcript_prefix_hash,
                         const std::uint64_t previous_input_position,
                         const std::uint64_t segment_input_count,
                         const std::uint64_t segment_input_hash,
                         const std::uint64_t segment_transcript_hash,
                         const std::uint64_t input_position) noexcept {
  std::uint64_t hash = BeginHash("rund.replay.checkpoint.transcript");
  node::replay_detail::Mix(hash, previous_transcript_prefix_hash);
  node::replay_detail::Mix(hash, previous_input_position);
  node::replay_detail::Mix(hash, segment_input_count);
  node::replay_detail::Mix(hash, segment_input_hash);
  node::replay_detail::Mix(hash, segment_transcript_hash);
  node::replay_detail::Mix(hash, input_position);
  return hash;
}

[[nodiscard]] std::uint64_t HashCheckpoint(
    const std::uint64_t segment_count, const std::uint64_t input_position,
    const std::uint64_t state_size, const std::uint64_t state_hash,
    const std::uint64_t boundary_hash, const std::uint64_t prefix_hash,
    const std::uint64_t transcript_prefix_hash) noexcept {
  std::uint64_t hash = BeginHash("rund.replay.checkpoint");
  node::replay_detail::Mix(hash, segment_count);
  node::replay_detail::Mix(hash, input_position);
  node::replay_detail::Mix(hash, state_size);
  node::replay_detail::Mix(hash, state_hash);
  node::replay_detail::Mix(hash, boundary_hash);
  node::replay_detail::Mix(hash, prefix_hash);
  node::replay_detail::Mix(hash, transcript_prefix_hash);
  return hash;
}

} // namespace

OwnedCheckpointState
OwnedCheckpointState::Copy(const std::uint64_t schema,
                           const std::span<const std::byte> source) {
  OwnedCheckpointState state{};
  state.bytes.reserve(source.size());
  state.hash = BeginHash("rund.replay.checkpoint.state");
  node::replay_detail::Mix(state.hash, schema);
  node::replay_detail::Mix(state.hash,
                           static_cast<std::uint64_t>(source.size()));
  for (const std::byte value : source) {
    state.bytes.push_back(value);
    node::replay_detail::Mix(state.hash, std::to_integer<std::uint8_t>(value));
  }
  return state;
}

OwnedCheckpointState
OwnedCheckpointState::Adopt(const std::uint64_t schema,
                            std::vector<std::byte> source) noexcept {
  const std::uint64_t hash = HashCheckpointState(schema, source);
  return OwnedCheckpointState{.bytes = std::move(source), .hash = hash};
}

Checkpoint::Data::Data(const std::uint64_t prior_segment_count,
                       const std::uint64_t prior_input_position,
                       const std::uint64_t prior_prefix_hash,
                       const std::uint64_t prior_transcript_prefix_hash,
                       const std::uint64_t record_hash,
                       const std::uint64_t record_input_count,
                       const std::uint64_t record_input_hash,
                       const std::uint64_t record_transcript_hash,
                       const std::uint64_t schema,
                       OwnedCheckpointState owned_state)
    : state(std::move(owned_state.bytes)), state_schema(schema),
      state_size(static_cast<std::uint64_t>(state.size())),
      state_hash(owned_state.hash), previous_segment_count(prior_segment_count),
      previous_input_position(prior_input_position),
      previous_prefix_hash(prior_prefix_hash),
      previous_transcript_prefix_hash(prior_transcript_prefix_hash),
      segment_record_hash(record_hash), segment_input_count(record_input_count),
      segment_input_hash(record_input_hash),
      segment_transcript_hash(record_transcript_hash) {
  if (!rund::kernel::checked::add(previous_segment_count, 1u, segment_count) ||
      !rund::kernel::checked::add(previous_input_position, segment_input_count,
                                  input_position)) {
    return;
  }
  boundary_hash = HashCheckpointBoundary(
      segment_count, segment_record_hash, segment_input_count,
      segment_input_hash, segment_transcript_hash);
  prefix_hash =
      HashCheckpointPrefix(previous_prefix_hash, segment_count, boundary_hash);
  transcript_prefix_hash = HashCheckpointTranscript(
      previous_transcript_prefix_hash, previous_input_position,
      segment_input_count, segment_input_hash, segment_transcript_hash,
      input_position);
  checkpoint_hash =
      HashCheckpoint(segment_count, input_position, state_size, state_hash,
                     boundary_hash, prefix_hash, transcript_prefix_hash);
}

bool Checkpoint::Data::valid() const noexcept {
  // Capture computes state_hash while creating the private snapshot; decode
  // computes it before adopting the decoded allocation. The immutable value
  // then validates its scalar chain in O(1).
  if (segment_count == 0u || state_schema == 0u ||
      state_size != static_cast<std::uint64_t>(state.size())) {
    return false;
  }
  std::uint64_t expected_segment_count = 0u;
  std::uint64_t expected_input_position = 0u;
  if (!rund::kernel::checked::add(previous_segment_count, 1u,
                                  expected_segment_count) ||
      expected_segment_count != segment_count ||
      !rund::kernel::checked::add(previous_input_position, segment_input_count,
                                  expected_input_position) ||
      expected_input_position != input_position) {
    return false;
  }
  if (previous_segment_count == 0u &&
      (previous_input_position != 0u || previous_prefix_hash != 0u ||
       previous_transcript_prefix_hash != 0u)) {
    return false;
  }
  const std::uint64_t expected_boundary = HashCheckpointBoundary(
      segment_count, segment_record_hash, segment_input_count,
      segment_input_hash, segment_transcript_hash);
  const std::uint64_t expected_prefix = HashCheckpointPrefix(
      previous_prefix_hash, segment_count, expected_boundary);
  const std::uint64_t expected_transcript = HashCheckpointTranscript(
      previous_transcript_prefix_hash, previous_input_position,
      segment_input_count, segment_input_hash, segment_transcript_hash,
      input_position);
  return boundary_hash == expected_boundary && prefix_hash == expected_prefix &&
         transcript_prefix_hash == expected_transcript &&
         checkpoint_hash == HashCheckpoint(segment_count, input_position,
                                           state_size, state_hash,
                                           boundary_hash, prefix_hash,
                                           transcript_prefix_hash);
}

bool Checkpoint::ok() const noexcept { return code() == Code::Ok; }

Code Checkpoint::code() const noexcept {
  if (data_) {
    return data_->valid() ? Code::Ok : Code::CheckpointInvalid;
  }
  return code_ == Code::Ok ? Code::CheckpointMovedFrom : code_;
}

std::string_view Checkpoint::error() const noexcept {
  return ::rund::replay::error(code());
}

std::uint64_t Checkpoint::segment_count() const noexcept {
  return ok() ? data_->segment_count : 0u;
}

std::uint64_t Checkpoint::input_position() const noexcept {
  return ok() ? data_->input_position : 0u;
}

std::uint64_t Checkpoint::schema() const noexcept {
  return ok() ? data_->state_schema : 0u;
}

std::size_t Checkpoint::state_size() const noexcept {
  return ok() ? data_->state.size() : 0u;
}

std::span<const std::byte> Checkpoint::state() const noexcept {
  return ok() ? std::span<const std::byte>{data_->state}
              : std::span<const std::byte>{};
}

std::uint64_t Checkpoint::state_hash() const noexcept {
  return ok() ? data_->state_hash : 0u;
}

std::uint64_t Checkpoint::boundary_hash() const noexcept {
  return ok() ? data_->boundary_hash : 0u;
}

std::uint64_t Checkpoint::prefix_hash() const noexcept {
  return ok() ? data_->prefix_hash : 0u;
}

std::uint64_t Checkpoint::transcript_prefix_hash() const noexcept {
  return ok() ? data_->transcript_prefix_hash : 0u;
}

std::uint64_t Checkpoint::hash() const noexcept {
  return ok() ? data_->checkpoint_hash : 0u;
}

Save Checkpoint::persist(const detail::Output output) const noexcept {
  if (!ok()) {
    return Save{code(), 0u, 0u};
  }
  const Data &data = *data_;
  node::replay_detail::artifact::Writer out{node::replay_detail::artifact::Sink{
      .state = output.state, .write = output.write}};
  static_cast<void>(
      out.header(node::replay_detail::artifact::Kind::Checkpoint) &&
      out.varuint(data.previous_segment_count) &&
      out.varuint(data.previous_input_position) &&
      out.fixed64(data.previous_prefix_hash) &&
      out.fixed64(data.previous_transcript_prefix_hash) &&
      out.fixed64(data.segment_record_hash) &&
      out.varuint(data.segment_input_count) &&
      out.fixed64(data.segment_input_hash) &&
      out.fixed64(data.segment_transcript_hash) &&
      out.varuint(data.state_schema) && out.varuint(data.state_size) &&
      out.fixed64(data.state_hash) && out.fixed64(data.checkpoint_hash) &&
      out.raw(data.state));
  const node::replay_detail::artifact::Result saved = out.finish();
  return Save{saved.code, saved.bytes, saved.writes};
}

Load<Checkpoint> Checkpoint::load(const std::span<const std::byte> artifact,
                                  const Limits limits) noexcept {
  if (artifact.size() > limits.max_bytes) {
    return Load<Checkpoint>{
        ::rund::replay::Code::CheckpointEncodedCapacityExceeded, std::nullopt};
  }
  if (limits.max_entries < 1u) {
    return Load<Checkpoint>{
        ::rund::replay::Code::CheckpointEntryCapacityExceeded, std::nullopt};
  }
  try {
    node::replay_detail::artifact::Reader in{artifact};
    if (!in.header(node::replay_detail::artifact::Kind::Checkpoint)) {
      return Load<Checkpoint>{::rund::replay::Code::CheckpointBadHeader,
                              std::nullopt};
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
      return Load<Checkpoint>{::rund::replay::Code::CheckpointBadField,
                              std::nullopt};
    }
    if (state_schema == 0u) {
      return Load<Checkpoint>{::rund::replay::Code::StateSchemaInvalid,
                              std::nullopt};
    }
    if (state_size > limits.max_state_bytes ||
        state_size > static_cast<std::uint64_t>(
                         std::numeric_limits<std::size_t>::max()) ||
        state_size != in.remaining()) {
      return Load<Checkpoint>{
          ::rund::replay::Code::CheckpointStateCapacityExceeded, std::nullopt};
    }
    std::span<const std::byte> state_view{};
    if (!in.take(static_cast<std::size_t>(state_size), state_view) ||
        !in.done()) {
      return Load<Checkpoint>{::rund::replay::Code::CheckpointBadField,
                              std::nullopt};
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
      return Load<Checkpoint>{
          ::rund::replay::Code::CheckpointStateAdoptionInvalid, std::nullopt};
    }
    if (candidate->state_schema != state_schema ||
        candidate->state_size != state_size) {
      return Load<Checkpoint>{::rund::replay::Code::CheckpointStateSizeInvalid,
                              std::nullopt};
    }
    if (candidate->state_hash != state_hash) {
      return Load<Checkpoint>{::rund::replay::Code::CheckpointStateHashInvalid,
                              std::nullopt};
    }
    if (!candidate->valid()) {
      return Load<Checkpoint>{::rund::replay::Code::CheckpointChainInvalid,
                              std::nullopt};
    }
    if (candidate->checkpoint_hash != checkpoint_hash) {
      return Load<Checkpoint>{::rund::replay::Code::CheckpointHashInvalid,
                              std::nullopt};
    }

    return Load<Checkpoint>{Code::Ok, Checkpoint{std::move(candidate)}};
  } catch (const std::bad_alloc &) {
    return Load<Checkpoint>{::rund::replay::Code::CheckpointCapacityExceeded,
                            std::nullopt};
  } catch (const std::length_error &) {
    return Load<Checkpoint>{::rund::replay::Code::CheckpointCapacityExceeded,
                            std::nullopt};
  } catch (...) {
    return Load<Checkpoint>{::rund::replay::Code::CheckpointLoadFailed,
                            std::nullopt};
  }
}

Resume Binding::resume(const Checkpoint &checkpoint) const noexcept {
  Code binding = code();
  if (binding == Code::Ok && !checkpointable_) {
    binding = Code::StateSchemaInvalid;
  }
  if (binding == Code::Ok) {
    binding = checkpoint.code();
  }
  if (binding == Code::Ok && checkpoint.schema() != schema_) {
    binding = Code::StateSchemaInvalid;
  }
  return Resume{checkpoint, restore_, binding};
}

} // namespace rund::replay
