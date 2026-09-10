#include "local.hpp"

#include <node/runtime/replay/hash.hpp>

#include <utility>

namespace rund::replay {

OwnedCheckpointState
OwnedCheckpointState::Copy(const std::uint64_t schema,
                           const std::span<const std::byte> source) {
  OwnedCheckpointState state{};
  state.bytes.reserve(source.size());
  state.hash = checkpoint_detail::begin_state_hash(
      schema, static_cast<std::uint64_t>(source.size()));
  for (const std::byte value : source) {
    state.bytes.push_back(value);
    node::replay_detail::Mix(state.hash, std::to_integer<std::uint8_t>(value));
  }
  return state;
}

OwnedCheckpointState
OwnedCheckpointState::Adopt(const std::uint64_t schema,
                            std::vector<std::byte> source) noexcept {
  const std::uint64_t hash = checkpoint_detail::hash_state(schema, source);
  return OwnedCheckpointState{.bytes = std::move(source), .hash = hash};
}

} // namespace rund::replay
