#include <node/runtime/replay/host/payload.hpp>

#include "backend.hpp"
#include "chunk.hpp"
#include "diagnostic/ring.hpp"
#include "hash.hpp"
#include <kernel/core/checked.hpp>

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace rund::node::replay_detail {
namespace {

[[nodiscard]] bool
IsKnownCodec(const ::rund::node::replay_detail::payload::Codec codec) noexcept {
  return codec == ::rund::node::replay_detail::payload::Codec::Raw ||
         codec == ::rund::node::replay_detail::payload::Codec::Rle;
}

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

[[nodiscard]] bool WholeChunk(
    const ::rund::node::replay_detail::payload::ArchiveRecord &record,
    const ::rund::node::replay_detail::payload::Archive &archive) noexcept {
  if (record.pieces.size() != 1u) {
    return false;
  }
  const ::rund::node::replay_detail::payload::ArchivePiece &piece =
      record.pieces.front();
  if (piece.chunk_id >= archive.chunks.size()) {
    return false;
  }
  const ::rund::node::replay_detail::payload::ArchiveChunk &chunk =
      archive.chunks[static_cast<std::size_t>(piece.chunk_id)];
  return piece.offset == 0u && piece.size == record.metadata.completed_bytes &&
         chunk.uncompressed_bytes == record.metadata.completed_bytes &&
         chunk.uncompressed_hash.value == record.metadata.payload_hash.value;
}

} // namespace

bool IsPayloadKind(const ::rund::host::EventKind kind) noexcept {
  return kind == ::rund::host::EventKind::IoRead ||
         kind == ::rund::host::EventKind::IoWrite;
}

bool EventRequiresPayload(const ::rund::host::Event &event) noexcept {
  return IsPayloadKind(event.kind) && event.status == ::rund::host::Status::Ok;
}

bool EventsRequirePayload(
    const std::vector<::rund::host::Event> &events) noexcept {
  for (const ::rund::host::Event &event : events) {
    if (EventRequiresPayload(event)) {
      return true;
    }
  }
  return false;
}

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

bool ValidArchive(
    const ::rund::node::replay_detail::payload::Archive &archive) {
  if (!payload::ValidDiagnosticArchive(archive.diagnostic)) {
    return false;
  }
  if (archive.records.empty()) {
    return archive.chunks.empty() && archive.payload_hash == 0u &&
           archive.storage.logical_bytes == 0u &&
           archive.storage.encoded_bytes == 0u &&
           archive.storage.retained_bytes == 0u &&
           archive.storage.copied_bytes == 0u &&
           archive.storage.physical_bytes == 0u &&
           archive.storage.allocated_bytes == 0u &&
           archive.storage.reserved_bytes == 0u &&
           archive.storage.growths == 0u && archive.storage.chunk_count == 0u &&
           archive.storage.segment_count == 0u &&
           archive.spill_generation == nullptr;
  }
  if (archive.payload_hash == 0u ||
      archive.chunks.size() >
          static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      (archive.storage.mode != ::rund::replay::StorageMode::Memory &&
       archive.storage.mode != ::rund::replay::StorageMode::Spill)) {
    return false;
  }

  const bool spilled =
      archive.storage.mode == ::rund::replay::StorageMode::Spill;
  if ((spilled && (archive.storage.retained_bytes != 0u ||
                   archive.storage.copied_bytes != 0u ||
                   archive.storage.reserved_bytes != 0u ||
                   archive.spill_generation == nullptr)) ||
      (!spilled &&
       (archive.storage.retained_bytes != archive.storage.encoded_bytes ||
        (archive.storage.copied_bytes != 0u &&
         archive.storage.copied_bytes != archive.storage.encoded_bytes) ||
        archive.storage.physical_bytes != 0u ||
        archive.storage.allocated_bytes != 0u ||
        archive.storage.reserved_bytes != 0u ||
        archive.spill_generation != nullptr))) {
    return false;
  }
  std::uint64_t encoded_bytes = 0u;
  std::vector<bool> referenced(archive.chunks.size(), false);
  for (std::size_t index = 0u; index < archive.chunks.size(); ++index) {
    const ::rund::node::replay_detail::payload::ArchiveChunk &chunk =
        archive.chunks[index];
    if (chunk.chunk_id != index || !IsKnownCodec(chunk.codec) ||
        chunk.uncompressed_bytes > payload::kChunkBytes ||
        chunk.encoded_bytes > payload::kChunkBytes ||
        chunk.spilled != spilled ||
        !rund::kernel::checked::add(encoded_bytes, chunk.encoded_bytes,
                                    encoded_bytes)) {
      return false;
    }
    if (!spilled) {
      if (chunk.encoded_bytes != chunk.encoded.size() ||
          chunk.segment_index != 0u || chunk.segment_offset != 0u ||
          chunk.segment_record_bytes != 0u) {
        return false;
      }
    } else {
      std::uint64_t segment_end = 0u;
      if (!chunk.encoded.empty() || chunk.segment_record_bytes == 0u ||
          chunk.segment_record_bytes > (2u * payload::kChunkBytes) ||
          chunk.segment_index >= archive.storage.segment_count ||
          !rund::kernel::checked::add(
              chunk.segment_offset, chunk.segment_record_bytes, segment_end)) {
        return false;
      }
    }
  }

  std::uint64_t logical_bytes = 0u;
  payload::ArchiveHash archive_hash{
      static_cast<std::uint64_t>(archive.records.size())};
  for (const ::rund::node::replay_detail::payload::ArchiveRecord &record :
       archive.records) {
    const payload::Record &metadata = record.metadata;
    const bool host_record =
        metadata.role == ::rund::node::replay_detail::payload::Role::Host;
    const bool input_record =
        metadata.role == ::rund::node::replay_detail::payload::Role::Input;
    std::uint64_t event_end = 0u;
    std::uint64_t payload_end = 0u;
    if ((!host_record && !input_record) ||
        (host_record &&
         (!IsPayloadKind(metadata.kind) || metadata.input_source != 0u ||
          metadata.input_schema != 0u || metadata.input_sequence != 0u ||
          metadata.source_event_offset != 0u ||
          metadata.source_event_count != 0u ||
          metadata.source_payload_offset != 0u ||
          metadata.source_payload_count != 0u || metadata.source_hash != 0u)) ||
        (input_record &&
         (metadata.event_sequence != 0u ||
          metadata.kind != ::rund::host::EventKind::None ||
          metadata.input_source == 0u || metadata.input_schema == 0u ||
          metadata.source_hash == 0u ||
          !rund::kernel::checked::add(metadata.source_event_offset,
                                      metadata.source_event_count, event_end) ||
          !rund::kernel::checked::add(metadata.source_payload_offset,
                                      metadata.source_payload_count,
                                      payload_end))) ||
        !rund::kernel::checked::add(logical_bytes, metadata.completed_bytes,
                                    logical_bytes)) {
      return false;
    }
    std::uint64_t piece_bytes = 0u;
    std::optional<payload::ByteHash> payload_hash{};
    if (!spilled && !WholeChunk(record, archive)) {
      payload_hash.emplace();
    }
    payload::RecordHash record_hash{
        metadata, metadata.completed_bytes,
        static_cast<std::uint64_t>(record.pieces.size())};

    for (const ::rund::node::replay_detail::payload::ArchivePiece &piece :
         record.pieces) {
      if (piece.chunk_id >= archive.chunks.size()) {
        return false;
      }
      const std::size_t chunk_index = static_cast<std::size_t>(piece.chunk_id);
      const ::rund::node::replay_detail::payload::ArchiveChunk &chunk =
          archive.chunks[chunk_index];
      if (piece.offset != 0u || piece.size != chunk.uncompressed_bytes ||
          !rund::kernel::checked::add(piece_bytes, piece.size, piece_bytes)) {
        return false;
      }
      record_hash.Append(chunk.uncompressed_hash, chunk.uncompressed_bytes);
      if (spilled) {
        referenced[chunk_index] = true;
        continue;
      }
      const bool first_reference = !referenced[chunk_index];
      const bool valid =
          first_reference
              ? payload::Verify(chunk.uncompressed_hash,
                                chunk.uncompressed_bytes, chunk.codec,
                                chunk.encoded.span(),
                                payload_hash ? &*payload_hash : nullptr)
              : !payload_hash || payload::AppendDecodedBytes(
                                     chunk.uncompressed_bytes, chunk.codec,
                                     chunk.encoded.span(), *payload_hash);
      if (!valid) {
        return false;
      }
      referenced[chunk_index] = true;
    }
    if (piece_bytes != metadata.completed_bytes ||
        (payload_hash &&
         payload_hash->Finish() != metadata.payload_hash.value)) {
      return false;
    }
    archive_hash.Append(record_hash.Finish());
  }

  for (const bool used : referenced) {
    if (!used) {
      return false;
    }
  }
  constexpr std::uint64_t kSpillRecordHeaderBytes =
      sizeof(std::uint64_t) * 5u + sizeof(std::uint8_t);
  const std::uint64_t chunk_count =
      static_cast<std::uint64_t>(archive.chunks.size());
  std::uint64_t header_bytes = 0u;
  std::uint64_t spill_physical_bytes = 0u;
  if (spilled && (!rund::kernel::checked::mul(
                      chunk_count, kSpillRecordHeaderBytes, header_bytes) ||
                  !rund::kernel::checked::add(encoded_bytes, header_bytes,
                                              spill_physical_bytes))) {
    return false;
  }
  return logical_bytes == archive.storage.logical_bytes &&
         encoded_bytes == archive.storage.encoded_bytes &&
         archive.storage.chunk_count == archive.chunks.size() &&
         (!spilled || (archive.storage.physical_bytes == spill_physical_bytes &&
                       archive.storage.allocated_bytes >=
                           archive.storage.physical_bytes)) &&
         (spilled || archive.storage.segment_count == 0u) &&
         archive_hash.Finish() == archive.payload_hash;
}

} // namespace rund::node::replay_detail
