#include <node/runtime/replay/host/payload.hpp>

#include "../hash.hpp"

#include <kernel/core/checked.hpp>

#include <optional>
#include <vector>

namespace rund::node::replay_detail {
namespace {

class ArchiveHostCursor final {
public:
  explicit ArchiveHostCursor(
      const ::rund::node::replay_detail::payload::Archive &archive) noexcept
      : archive_(archive) {}

  [[nodiscard]] bool Limit(const std::size_t physical_limit) noexcept {
    if (physical_limit < physical_ ||
        physical_limit > archive_.records.size()) {
      return false;
    }
    physical_limit_ = physical_limit;
    return true;
  }

  [[nodiscard]] bool Seek(const std::uint64_t offset) noexcept {
    if (offset < logical_) {
      return false;
    }
    while (logical_ < offset) {
      if (!Next().has_value()) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] std::optional<payload::SourcePayloadBinding> Next() noexcept {
    while (physical_ < physical_limit_) {
      const ::rund::node::replay_detail::payload::ArchiveRecord &record =
          archive_.records[physical_++];
      const payload::Record &metadata = record.metadata;
      if (metadata.role != ::rund::node::replay_detail::payload::Role::Host) {
        continue;
      }
      ++logical_;
      return payload::SourcePayloadBinding{
          .event_sequence = metadata.event_sequence,
          .kind = metadata.kind,
          .completed_bytes = metadata.completed_bytes,
          .payload_hash = metadata.payload_hash};
    }
    return std::nullopt;
  }

private:
  const ::rund::node::replay_detail::payload::Archive &archive_;
  std::size_t physical_ = 0u;
  std::size_t physical_limit_ = 0u;
  std::uint64_t logical_ = 0u;
};

[[nodiscard]] bool SourceRangeHash(
    const std::vector<::rund::host::Event> &events,
    const ::rund::node::replay_detail::payload::ArchiveRecord &input,
    const std::size_t input_record_index, ArchiveHostCursor &payloads,
    std::uint64_t &event_end, std::uint64_t &out) noexcept {
  const payload::Record &metadata = input.metadata;
  if (metadata.source_event_offset > events.size() ||
      metadata.source_event_count >
          events.size() - metadata.source_event_offset ||
      metadata.source_event_offset < event_end ||
      !payloads.Limit(input_record_index) ||
      !payloads.Seek(metadata.source_payload_offset)) {
    return false;
  }
  const std::size_t event_begin =
      static_cast<std::size_t>(metadata.source_event_offset);
  const std::span<const ::rund::host::Event> source_events =
      std::span<const ::rund::host::Event>{events}.subspan(
          event_begin, static_cast<std::size_t>(metadata.source_event_count));
  const auto next_payload = [&payloads]() noexcept { return payloads.Next(); };
  const std::optional<std::uint64_t> source_hash =
      payload::ComputeSourceRangeHash(
          metadata.source_event_offset, source_events,
          metadata.source_payload_offset, metadata.source_payload_count,
          next_payload);
  if (!source_hash.has_value() ||
      !rund::kernel::checked::add(metadata.source_event_offset,
                                  metadata.source_event_count, event_end)) {
    return false;
  }
  out = *source_hash;
  return true;
}

} // namespace

::rund::replay::Code BindPayloads(
    const std::vector<::rund::host::Event> &events,
    const ::rund::node::replay_detail::payload::Archive &archive) noexcept {
  ArchiveHostCursor source_payloads{archive};
  std::uint64_t source_event_end = 0u;
  std::size_t host_event_index = 0u;
  std::uint64_t last_host_payload_sequence = 0u;
  for (std::size_t record_index = 0u; record_index < archive.records.size();
       ++record_index) {
    const ::rund::node::replay_detail::payload::ArchiveRecord &payload =
        archive.records[record_index];
    const auto &metadata = payload.metadata;
    if (metadata.role == ::rund::node::replay_detail::payload::Role::Input) {
      std::uint64_t source_hash = 0u;
      if (!SourceRangeHash(events, payload, record_index, source_payloads,
                           source_event_end, source_hash) ||
          source_hash != metadata.source_hash) {
        return ::rund::replay::Code::InputSourceHashMismatch;
      }
      continue;
    }
    if (metadata.event_sequence <= last_host_payload_sequence) {
      return ::rund::replay::Code::HostDuplicateField;
    }
    while (host_event_index < events.size() &&
           events[host_event_index].sequence < metadata.event_sequence) {
      if (EventRequiresPayload(events[host_event_index])) {
        return ::rund::replay::Code::HostPayloadMissing;
      }
      ++host_event_index;
    }
    if (host_event_index == events.size() ||
        events[host_event_index].sequence != metadata.event_sequence) {
      return ::rund::replay::Code::HostPayloadMissing;
    }
    const ::rund::host::Event &event = events[host_event_index];
    if (!EventRequiresPayload(event) || event.kind != metadata.kind ||
        event.completed_bytes != metadata.completed_bytes ||
        event.payload_hash.value != metadata.payload_hash.value) {
      return ::rund::replay::Code::HostPayloadHashInvalid;
    }
    last_host_payload_sequence = metadata.event_sequence;
    ++host_event_index;
  }
  while (host_event_index < events.size()) {
    if (EventRequiresPayload(events[host_event_index])) {
      return ::rund::replay::Code::HostPayloadMissing;
    }
    ++host_event_index;
  }
  std::size_t diagnostic_event_index = 0u;
  std::uint64_t last_diagnostic_sequence = 0u;
  for (const ::rund::node::replay_detail::payload::DiagnosticRecord
           &diagnostic : archive.diagnostic.records) {
    if (diagnostic.event_sequence <= last_diagnostic_sequence) {
      return ::rund::replay::Code::HostDiagnosticEventMismatch;
    }
    while (diagnostic_event_index < events.size() &&
           events[diagnostic_event_index].sequence <
               diagnostic.event_sequence) {
      ++diagnostic_event_index;
    }
    if (diagnostic_event_index == events.size()) {
      return ::rund::replay::Code::HostDiagnosticEventMismatch;
    }
    const ::rund::host::Event &event = events[diagnostic_event_index];
    if (event.sequence != diagnostic.event_sequence ||
        event.kind != diagnostic.kind ||
        event.status != ::rund::host::Status::Ok ||
        event.completed_bytes != diagnostic.byte_count ||
        event.payload_hash.value != diagnostic.payload_hash.value) {
      return ::rund::replay::Code::HostDiagnosticEventMismatch;
    }
    last_diagnostic_sequence = diagnostic.event_sequence;
    ++diagnostic_event_index;
  }
  return ::rund::replay::Code::Ok;
}

} // namespace rund::node::replay_detail
