#pragma once

#include <node/runtime/replay/host/bytes.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace rund::node::replay_detail {

struct InputPatch final {
  std::uint64_t source = 0u;
  std::uint64_t schema = 0u;
  std::uint64_t sequence = 0u;
  std::uint64_t payload_hash = 0u;
  std::size_t offset = 0u;
  std::size_t size = 0u;
};

// Source-private immutable bridge between the public scenario facade and the
// Scheduler. SessionConfig exposes only an incomplete shared owner, so callers
// cannot manufacture rows, hashes, or a competing transcript authority.
class InputPlan final {
public:
  explicit InputPlan(::rund::node::replay_detail::payload::Bytes bytes,
                     std::vector<InputPatch> patches) noexcept
      : bytes_(std::move(bytes)), patches_(std::move(patches)) {}

  [[nodiscard]] const InputPatch *
  find(const std::uint64_t source, const std::uint64_t schema,
       const std::uint64_t sequence) const noexcept {
    const auto found = std::lower_bound(
        patches_.begin(), patches_.end(), source,
        [schema, sequence](const InputPatch &patch,
                           const std::uint64_t candidate_source) {
          if (patch.source != candidate_source) {
            return patch.source < candidate_source;
          }
          if (patch.schema != schema) {
            return patch.schema < schema;
          }
          return patch.sequence < sequence;
        });
    return found != patches_.end() && found->source == source &&
                   found->schema == schema && found->sequence == sequence
               ? &*found
               : nullptr;
  }

  [[nodiscard]] ::rund::node::replay_detail::payload::Bytes
  bytes(const InputPatch &patch) const noexcept {
    return bytes_.slice(patch.offset, patch.size);
  }

  [[nodiscard]] std::size_t retained_bytes() const noexcept {
    return bytes_.retained_bytes();
  }

private:
  ::rund::node::replay_detail::payload::Bytes bytes_{};
  std::vector<InputPatch> patches_{};
};

} // namespace rund::node::replay_detail
