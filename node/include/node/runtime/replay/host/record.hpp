#pragma once

#include <rund/host/event.hpp>

#include <cstdint>

namespace rund::node::replay_detail::payload {

// The replay archive has one byte owner. Host I/O and caller-declared opaque
// inputs are record roles within that owner, not parallel blob stores.
enum class Role : std::uint8_t {
  Host = 0u,
  Input = 1u,
};

// Sole owner of persisted replay-record semantics. Archive, live storage, and
// materialized representations compose this value and only add their
// representation-specific byte ownership or indexing state.
struct Record final {
  Role role = Role::Host;
  std::uint64_t event_sequence = 0u;
  ::rund::host::EventKind kind = ::rund::host::EventKind::None;
  std::uint64_t input_source = 0u;
  std::uint64_t input_schema = 0u;
  std::uint64_t input_sequence = 0u;
  std::uint64_t source_event_offset = 0u;
  std::uint64_t source_event_count = 0u;
  std::uint64_t source_payload_offset = 0u;
  std::uint64_t source_payload_count = 0u;
  std::uint64_t source_hash = 0u;
  std::uint64_t completed_bytes = 0u;
  ::rund::StableHash payload_hash{};

  [[nodiscard]] friend constexpr bool operator==(const Record &left,
                                                 const Record &right) noexcept {
    return left.role == right.role &&
           left.event_sequence == right.event_sequence &&
           left.kind == right.kind && left.input_source == right.input_source &&
           left.input_schema == right.input_schema &&
           left.input_sequence == right.input_sequence &&
           left.source_event_offset == right.source_event_offset &&
           left.source_event_count == right.source_event_count &&
           left.source_payload_offset == right.source_payload_offset &&
           left.source_payload_count == right.source_payload_count &&
           left.source_hash == right.source_hash &&
           left.completed_bytes == right.completed_bytes &&
           left.payload_hash.value == right.payload_hash.value;
  }
};

} // namespace rund::node::replay_detail::payload
