#pragma once

#include <rund/replay/state.hpp>

#include <cstdint>
#include <span>
#include <vector>

namespace rund::replay {

struct OwnedCheckpointState final {
  [[nodiscard]] static OwnedCheckpointState
  Copy(std::uint64_t schema, std::span<const std::byte> source);
  [[nodiscard]] static OwnedCheckpointState
  Adopt(std::uint64_t schema, std::vector<std::byte> source) noexcept;

  std::vector<std::byte> bytes{};
  std::uint64_t hash = 0u;
};

struct Checkpoint::Data {
  Data(std::uint64_t prior_segment_count, std::uint64_t prior_input_position,
       std::uint64_t prior_prefix_hash,
       std::uint64_t prior_transcript_prefix_hash, std::uint64_t record_hash,
       std::uint64_t record_input_count, std::uint64_t record_input_hash,
       std::uint64_t record_transcript_hash, std::uint64_t schema,
       OwnedCheckpointState owned_state);

  [[nodiscard]] bool valid() const noexcept;

  std::vector<std::byte> state{};
  std::uint64_t state_schema = 0u;
  std::uint64_t state_size = 0u;
  std::uint64_t state_hash = 0u;
  std::uint64_t previous_segment_count = 0u;
  std::uint64_t previous_input_position = 0u;
  std::uint64_t previous_prefix_hash = 0u;
  std::uint64_t previous_transcript_prefix_hash = 0u;
  std::uint64_t segment_count = 0u;
  std::uint64_t segment_record_hash = 0u;
  std::uint64_t segment_input_count = 0u;
  std::uint64_t segment_input_hash = 0u;
  std::uint64_t segment_transcript_hash = 0u;
  std::uint64_t input_position = 0u;
  std::uint64_t boundary_hash = 0u;
  std::uint64_t prefix_hash = 0u;
  std::uint64_t transcript_prefix_hash = 0u;
  std::uint64_t checkpoint_hash = 0u;
};

} // namespace rund::replay
