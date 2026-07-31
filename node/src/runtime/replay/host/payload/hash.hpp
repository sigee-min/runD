#pragma once

#include <kernel/core/checked.hpp>

#include "../../../../host/hash/bytes.hpp"
#include "../../../../host/hash/fields.hpp"

#include <node/runtime/replay/host/archive.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <span>

namespace rund::node::replay_detail::payload {

class Capture final {
public:
  [[nodiscard]] static Capture
  read(const std::span<const std::byte> bytes) noexcept {
    if (bytes.data() == nullptr && !bytes.empty()) {
      return {};
    }
    return Capture{bytes, ::rund::host::hash_bytes(bytes.data(), bytes.size())};
  }

  [[nodiscard]] static Capture
  verify(const std::span<const std::byte> bytes,
         const ::rund::StableHash expected) noexcept {
    Capture capture = read(bytes);
    if (!capture || capture.hash_.value != expected.value) {
      return {};
    }
    return capture;
  }

  [[nodiscard]] explicit operator bool() const noexcept { return valid_; }

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return valid_ ? bytes_ : std::span<const std::byte>{};
  }

  [[nodiscard]] ::rund::StableHash hash() const noexcept {
    return valid_ ? hash_ : ::rund::StableHash{};
  }

private:
  Capture() noexcept = default;

  Capture(const std::span<const std::byte> bytes,
          const ::rund::StableHash hash) noexcept
      : bytes_{bytes}, hash_{hash}, valid_{true} {}

  std::span<const std::byte> bytes_{};
  ::rund::StableHash hash_{};
  bool valid_ = false;
};

class ByteHash {
public:
  ByteHash() noexcept = default;

  void Append(const std::span<const std::byte> bytes) noexcept {
    hash_.Append(bytes);
  }

  [[nodiscard]] std::uint64_t Finish() const noexcept {
    return hash_.Finish().value;
  }

private:
  host_detail::StableByteHasher hash_{};
};

// One canonical identity recipe for the root host-event and host-payload range
// consumed by an input producer.
class SourceRangeHasher final {
public:
  constexpr SourceRangeHasher(const std::uint64_t event_offset,
                              const std::uint64_t event_count,
                              const std::uint64_t payload_offset,
                              const std::uint64_t payload_count) noexcept {
    hash_.Mix(0x72756e642e737263ull);
    hash_.Mix(event_offset);
    hash_.Mix(event_count);
    hash_.Mix(payload_offset);
    hash_.Mix(payload_count);
  }

  constexpr void AppendEvent(const ::rund::StableHash event_hash) noexcept {
    hash_.Mix(event_hash.value);
  }

  constexpr void AppendPayload(const std::uint64_t event_sequence,
                               const ::rund::host::EventKind kind,
                               const std::uint64_t completed_bytes,
                               const ::rund::StableHash payload_hash) noexcept {
    hash_.Mix(event_sequence);
    hash_.Mix(static_cast<std::uint16_t>(kind));
    hash_.Mix(completed_bytes);
    hash_.Mix(payload_hash.value);
  }

  [[nodiscard]] constexpr std::uint64_t Finish() const noexcept {
    return hash_.Finish().value;
  }

private:
  host_detail::StableHashState hash_{};
};

struct SourcePayloadBinding final {
  std::uint64_t event_sequence = 0u;
  ::rund::host::EventKind kind = ::rund::host::EventKind::None;
  std::uint64_t completed_bytes = 0u;
  ::rund::StableHash payload_hash{};
};

// Owns source range order, matching, and hashing for both the live Store and
// immutable archive. next_payload returns the next Host record in canonical
// order starting at payload_offset; it may index a Store or scan an archive,
// but cannot redefine the identity or matching contract. The two semantic
// cursors only advance, so work is O(E + P) with no allocation.
template <typename NextPayload>
[[nodiscard]] std::optional<std::uint64_t>
ComputeSourceRangeHash(const std::uint64_t event_offset,
                       const std::span<const ::rund::host::Event> events,
                       const std::uint64_t payload_offset,
                       const std::uint64_t payload_count,
                       NextPayload &&next_payload) noexcept {
  if (!rund::kernel::checked::add(payload_offset, payload_count)) {
    return std::nullopt;
  }
  SourceRangeHasher hash{event_offset,
                         static_cast<std::uint64_t>(events.size()),
                         payload_offset, payload_count};
  std::uint64_t expected_sequence = event_offset;
  for (const ::rund::host::Event &event : events) {
    std::uint64_t next = 0u;
    if (!rund::kernel::checked::add(expected_sequence, 1u, next) ||
        event.task_id != 0u || event.sequence != next) {
      return std::nullopt;
    }
    expected_sequence = next;
    hash.AppendEvent(::rund::host::hash_event(event));
  }
  std::size_t event_index = 0u;
  for (std::uint64_t index = 0u; index < payload_count; ++index) {
    const std::optional<SourcePayloadBinding> payload = next_payload();
    if (!payload.has_value()) {
      return std::nullopt;
    }
    while (event_index < events.size() &&
           events[event_index].sequence < payload->event_sequence) {
      ++event_index;
    }
    if (event_index == events.size()) {
      return std::nullopt;
    }
    const ::rund::host::Event &event = events[event_index];
    if (event.sequence != payload->event_sequence ||
        event.kind != payload->kind ||
        event.completed_bytes != payload->completed_bytes ||
        event.payload_hash.value != payload->payload_hash.value) {
      return std::nullopt;
    }
    ++event_index;
    hash.AppendPayload(payload->event_sequence, payload->kind,
                       payload->completed_bytes, payload->payload_hash);
  }
  return hash.Finish();
}

// Canonical record framing includes the role and both identity domains. The
// unused identity domain is zero, so Host and Input rows cannot collide even
// when their bytes and numeric identifiers are equal.
class RecordHash {
public:
  constexpr RecordHash(
      const ::rund::node::replay_detail::payload::Record &record,
      const std::uint64_t materialized_bytes,
      const std::uint64_t piece_count) noexcept {
    hash_.Mix(static_cast<std::uint8_t>(record.role));
    hash_.Mix(record.event_sequence);
    hash_.Mix(static_cast<std::uint16_t>(record.kind));
    hash_.Mix(record.input_source);
    hash_.Mix(record.input_schema);
    hash_.Mix(record.input_sequence);
    if (record.role == ::rund::node::replay_detail::payload::Role::Input) {
      hash_.Mix(record.source_event_offset);
      hash_.Mix(record.source_event_count);
      hash_.Mix(record.source_payload_offset);
      hash_.Mix(record.source_payload_count);
      hash_.Mix(record.source_hash);
    }
    hash_.Mix(record.completed_bytes);
    hash_.Mix(record.payload_hash.value);
    hash_.Mix(materialized_bytes);
    hash_.Mix(piece_count);
  }

  constexpr void Append(const ::rund::StableHash chunk_hash,
                        const std::uint64_t chunk_bytes) noexcept {
    hash_.Mix(chunk_hash.value);
    hash_.Mix(chunk_bytes);
  }

  [[nodiscard]] constexpr std::uint64_t Finish() const noexcept {
    return hash_.Finish().value;
  }

private:
  host_detail::StableHashState hash_{};
};

class ArchiveHash {
public:
  explicit constexpr ArchiveHash(const std::uint64_t record_count) noexcept {
    hash_.Mix(record_count);
  }

  constexpr void Append(const std::uint64_t record_hash) noexcept {
    hash_.Mix(record_hash);
  }

  [[nodiscard]] constexpr std::uint64_t Finish() const noexcept {
    return hash_.Finish().value;
  }

private:
  host_detail::StableHashState hash_{};
};

} // namespace rund::node::replay_detail::payload
