#include "local.hpp"

#include <node/runtime/replay/hash.hpp>

#include <string_view>

namespace rund::replay::checkpoint_detail {
namespace {

[[nodiscard]] std::uint64_t begin_hash(const std::string_view domain) noexcept {
  std::uint64_t hash = node::replay_detail::kFnvOffset;
  node::replay_detail::MixString(hash, domain);
  return hash;
}

} // namespace

std::uint64_t begin_state_hash(const std::uint64_t schema,
                               const std::uint64_t state_size) noexcept {
  std::uint64_t hash = begin_hash("rund.replay.checkpoint.state");
  node::replay_detail::Mix(hash, schema);
  node::replay_detail::Mix(hash, state_size);
  return hash;
}

std::uint64_t hash_state(const std::uint64_t schema,
                         const std::span<const std::byte> state) noexcept {
  std::uint64_t hash =
      begin_state_hash(schema, static_cast<std::uint64_t>(state.size()));
  for (const std::byte value : state) {
    node::replay_detail::Mix(hash, std::to_integer<std::uint8_t>(value));
  }
  return hash;
}

std::uint64_t hash_boundary(const std::uint64_t segment_count,
                            const std::uint64_t record_hash,
                            const std::uint64_t input_count,
                            const std::uint64_t input_hash,
                            const std::uint64_t transcript_hash) noexcept {
  std::uint64_t hash = begin_hash("rund.replay.checkpoint.boundary");
  node::replay_detail::Mix(hash, segment_count);
  node::replay_detail::Mix(hash, record_hash);
  node::replay_detail::Mix(hash, input_count);
  node::replay_detail::Mix(hash, input_hash);
  node::replay_detail::Mix(hash, transcript_hash);
  return hash;
}

std::uint64_t hash_prefix(const std::uint64_t previous_prefix_hash,
                          const std::uint64_t segment_count,
                          const std::uint64_t boundary_hash) noexcept {
  std::uint64_t hash = begin_hash("rund.replay.checkpoint.prefix");
  node::replay_detail::Mix(hash, previous_prefix_hash);
  node::replay_detail::Mix(hash, segment_count);
  node::replay_detail::Mix(hash, boundary_hash);
  return hash;
}

std::uint64_t
hash_transcript(const std::uint64_t previous_transcript_prefix_hash,
                const std::uint64_t previous_input_position,
                const std::uint64_t segment_input_count,
                const std::uint64_t segment_input_hash,
                const std::uint64_t segment_transcript_hash,
                const std::uint64_t input_position) noexcept {
  std::uint64_t hash = begin_hash("rund.replay.checkpoint.transcript");
  node::replay_detail::Mix(hash, previous_transcript_prefix_hash);
  node::replay_detail::Mix(hash, previous_input_position);
  node::replay_detail::Mix(hash, segment_input_count);
  node::replay_detail::Mix(hash, segment_input_hash);
  node::replay_detail::Mix(hash, segment_transcript_hash);
  node::replay_detail::Mix(hash, input_position);
  return hash;
}

std::uint64_t hash_checkpoint(
    const std::uint64_t segment_count, const std::uint64_t input_position,
    const std::uint64_t state_size, const std::uint64_t state_hash,
    const std::uint64_t boundary_hash, const std::uint64_t prefix_hash,
    const std::uint64_t transcript_prefix_hash) noexcept {
  std::uint64_t hash = begin_hash("rund.replay.checkpoint");
  node::replay_detail::Mix(hash, segment_count);
  node::replay_detail::Mix(hash, input_position);
  node::replay_detail::Mix(hash, state_size);
  node::replay_detail::Mix(hash, state_hash);
  node::replay_detail::Mix(hash, boundary_hash);
  node::replay_detail::Mix(hash, prefix_hash);
  node::replay_detail::Mix(hash, transcript_prefix_hash);
  return hash;
}

} // namespace rund::replay::checkpoint_detail
