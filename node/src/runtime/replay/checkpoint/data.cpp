#include "local.hpp"

#include <kernel/core/checked.hpp>

#include <utility>

namespace rund::replay {

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
  boundary_hash = checkpoint_detail::hash_boundary(
      segment_count, segment_record_hash, segment_input_count,
      segment_input_hash, segment_transcript_hash);
  prefix_hash = checkpoint_detail::hash_prefix(previous_prefix_hash,
                                               segment_count, boundary_hash);
  transcript_prefix_hash = checkpoint_detail::hash_transcript(
      previous_transcript_prefix_hash, previous_input_position,
      segment_input_count, segment_input_hash, segment_transcript_hash,
      input_position);
  checkpoint_hash = checkpoint_detail::hash_checkpoint(
      segment_count, input_position, state_size, state_hash, boundary_hash,
      prefix_hash, transcript_prefix_hash);
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
  const std::uint64_t expected_boundary = checkpoint_detail::hash_boundary(
      segment_count, segment_record_hash, segment_input_count,
      segment_input_hash, segment_transcript_hash);
  const std::uint64_t expected_prefix = checkpoint_detail::hash_prefix(
      previous_prefix_hash, segment_count, expected_boundary);
  const std::uint64_t expected_transcript = checkpoint_detail::hash_transcript(
      previous_transcript_prefix_hash, previous_input_position,
      segment_input_count, segment_input_hash, segment_transcript_hash,
      input_position);
  return boundary_hash == expected_boundary && prefix_hash == expected_prefix &&
         transcript_prefix_hash == expected_transcript &&
         checkpoint_hash == checkpoint_detail::hash_checkpoint(
                                segment_count, input_position, state_size,
                                state_hash, boundary_hash, prefix_hash,
                                transcript_prefix_hash);
}

} // namespace rund::replay
