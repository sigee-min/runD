#include "local.hpp"

#include "../host/payload/backend.hpp"
#include "../host/payload/chunk.hpp"
#include "../host/payload/hash.hpp"
#include "../host/payload/store.hpp"

#include <node/runtime/replay/host/archive.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <utility>

namespace rund::node::replay_detail::artifact {
namespace {

[[nodiscard]] constexpr std::uint16_t
record_field(const bool present, const RecordField field) noexcept {
  return present ? static_cast<std::uint16_t>(field) : std::uint16_t{0u};
}

[[nodiscard]] bool write_chunk(Writer &out, const payload::ArchiveChunk &chunk,
                               const std::size_t index,
                               const payload::Store *const payloads) {
  if (!out.ok()) {
    return false;
  }
  if (chunk.chunk_id != index ||
      (chunk.codec != payload::Codec::Raw &&
       chunk.codec != payload::Codec::Rle) ||
      chunk.uncompressed_bytes > payload::kChunkBytes ||
      chunk.encoded_bytes > payload::kChunkBytes) {
    return out.reject(::rund::replay::Code::CodecInvariantInvalid);
  }
  payload::Bytes retained{};
  std::span<const std::byte> encoded = chunk.encoded.span();
  if (encoded.size() != chunk.encoded_bytes) {
    if (payloads == nullptr) {
      return out.reject(::rund::replay::Code::ArtifactPayloadMissing);
    }
    payload::EncodedResult loaded = payloads->Encoded(index);
    if (!loaded.ok()) {
      return out.reject(loaded.code);
    }
    retained = std::move(loaded.bytes);
    encoded = retained.span();
  }
  if (encoded.size() != chunk.encoded_bytes) {
    return out.reject(::rund::replay::Code::ArtifactPayloadMissing);
  }
  return out.u8(static_cast<std::uint8_t>(chunk.codec)) &&
         out.varuint(chunk.uncompressed_bytes) &&
         out.varuint(chunk.encoded_bytes) &&
         out.fixed64(chunk.uncompressed_hash.value) && out.raw(encoded);
}

[[nodiscard]] bool read_chunk(Reader &in, Admission &admission,
                              const std::size_t index,
                              payload::ArchiveChunk &chunk,
                              std::uint64_t &encoded_total,
                              std::uint64_t &uncompressed_total) {
  std::uint8_t codec = 0u;
  std::uint64_t encoded_bytes = 0u;
  std::size_t encoded_size = 0u;
  if (!in.u8(codec) || codec > static_cast<std::uint8_t>(payload::Codec::Rle) ||
      !in.varuint(chunk.uncompressed_bytes) ||
      chunk.uncompressed_bytes > payload::kChunkBytes ||
      !admission.payload(uncompressed_total, chunk.uncompressed_bytes) ||
      !in.varuint(encoded_bytes) || encoded_bytes > payload::kChunkBytes ||
      !admission.payload(encoded_total, encoded_bytes) ||
      !size(encoded_bytes, encoded_size) ||
      !in.fixed64(chunk.uncompressed_hash.value) ||
      in.position() > std::numeric_limits<std::uint64_t>::max()) {
    return false;
  }
  chunk.segment_offset = static_cast<std::uint64_t>(in.position());
  std::span<const std::byte> encoded{};
  if (!in.take(encoded_size, encoded)) {
    return false;
  }
  chunk.chunk_id = static_cast<std::uint64_t>(index);
  chunk.codec = static_cast<payload::Codec>(codec);
  chunk.encoded_bytes = encoded_bytes;
  return true;
}

[[nodiscard]] bool next_sequence(const std::uint64_t previous,
                                 const bool starts_at_zero,
                                 const std::uint64_t delta,
                                 std::uint64_t &result) noexcept {
  std::uint64_t base = 0u;
  if (!starts_at_zero && !rund::kernel::checked::add(previous, 1u, base)) {
    return false;
  }
  return rund::kernel::checked::add(base, delta, result);
}

// Schema 1 chunk deltas inhabit Z/(2^64). Every decoded representative is
// admitted against archive.chunks.size(). Keeping the modular projection
// explicit preserves the established bytes for both increasing and decreasing
// deduplicated chunk references without introducing a second grammar.
[[nodiscard]] constexpr std::uint64_t
chunk_delta(const std::uint64_t chunk, const std::uint64_t previous) noexcept {
  return chunk - previous;
}

[[nodiscard]] constexpr std::uint64_t
chunk_id(const std::uint64_t previous, const std::uint64_t delta) noexcept {
  return previous + delta;
}

} // namespace

bool write_payload(Writer &out, const payload::Archive &archive,
                   const payload::Store *const payloads,
                   const bool required_by_host_events) {
  if (!out.ok()) {
    return false;
  }
  const bool present = !archive.records.empty() ||
                       archive.diagnostic.hash != 0u || required_by_host_events;
  if (!out.u8(static_cast<std::uint8_t>(present))) {
    return false;
  }
  if (!present) {
    return true;
  }
  if (!out.fixed64(archive.payload_hash) ||
      !out.varuint(static_cast<std::uint64_t>(archive.chunks.size()))) {
    return false;
  }
  for (std::size_t index = 0u; index < archive.chunks.size(); ++index) {
    if (!write_chunk(out, archive.chunks[index], index, payloads)) {
      return false;
    }
  }
  if (!out.varuint(static_cast<std::uint64_t>(archive.records.size()))) {
    return false;
  }
  std::uint64_t previous_host_sequence = 0u;
  std::uint64_t previous_input_source = 0u;
  std::uint64_t previous_input_schema = 0u;
  std::uint64_t previous_input_sequence = 0u;
  bool first_input = true;
  std::uint64_t expected_event_offset = 0u;
  std::uint64_t expected_payload_offset = 0u;
  std::uint64_t previous_chunk = 0u;
  for (const payload::ArchiveRecord &record : archive.records) {
    const payload::Record &metadata = record.metadata;
    if (metadata.role != payload::Role::Host &&
        metadata.role != payload::Role::Input) {
      return out.reject(::rund::replay::Code::CodecInvariantInvalid);
    }
    const bool input = metadata.role == payload::Role::Input;
    std::uint64_t expected_sequence = 0u;
    if (!next_sequence(input ? previous_input_sequence : previous_host_sequence,
                       input && first_input, 0u, expected_sequence)) {
      return out.reject(::rund::replay::Code::CodecInvariantInvalid);
    }
    const bool source_changed =
        input && metadata.input_source != previous_input_source;
    const bool schema_changed =
        input && metadata.input_schema != previous_input_schema;
    const bool sequence_exceptional =
        (input ? metadata.input_sequence : metadata.event_sequence) !=
        expected_sequence;
    const bool event_range =
        input && (metadata.source_event_offset != expected_event_offset ||
                  metadata.source_event_count != 0u);
    const bool payload_range =
        input && (metadata.source_payload_offset != expected_payload_offset ||
                  metadata.source_payload_count != 0u);
    const bool source_hash_stored = input && (event_range || payload_range);
    const bool piece_count_exceptional = record.pieces.size() != 1u;
    const bool single_piece = !piece_count_exceptional;
    if (single_piece &&
        record.pieces.front().chunk_id >= archive.chunks.size()) {
      return out.reject(::rund::replay::Code::CodecInvariantInvalid);
    }
    const bool chunk_changed =
        single_piece && record.pieces.front().chunk_id != previous_chunk;
    const bool payload_hash_stored =
        piece_count_exceptional ||
        archive.chunks[static_cast<std::size_t>(record.pieces.front().chunk_id)]
                .uncompressed_hash.value != metadata.payload_hash.value;
    std::uint16_t fields = record_field(input, InputRole);
    fields |= record_field(source_changed, SourceChanged);
    fields |= record_field(schema_changed, SchemaChanged);
    fields |= record_field(sequence_exceptional, SequenceExceptional);
    fields |= record_field(event_range, EventRangePresent);
    fields |= record_field(payload_range, PayloadRangePresent);
    fields |= record_field(source_hash_stored, SourceHashStored);
    fields |= record_field(piece_count_exceptional, PieceCountExceptional);
    fields |= record_field(chunk_changed, ChunkChanged);
    fields |= record_field(payload_hash_stored, PayloadHashStored);
    if (!out.varuint(fields)) {
      return false;
    }
    if (metadata.role == payload::Role::Host) {
      std::uint64_t sequence_delta = 0u;
      if (sequence_exceptional) {
        if (!rund::kernel::checked::sub(metadata.event_sequence,
                                        expected_sequence, sequence_delta)) {
          return out.reject(::rund::replay::Code::CodecInvariantInvalid);
        }
        if (!out.varuint(sequence_delta)) {
          return false;
        }
      }
      previous_host_sequence = metadata.event_sequence;
    } else {
      std::uint64_t sequence_delta = 0u;
      std::uint64_t event_delta = 0u;
      std::uint64_t payload_delta = 0u;
      if ((sequence_exceptional &&
           !rund::kernel::checked::sub(metadata.input_sequence,
                                       expected_sequence, sequence_delta)) ||
          (event_range &&
           !rund::kernel::checked::sub(metadata.source_event_offset,
                                       expected_event_offset, event_delta)) ||
          (payload_range && !rund::kernel::checked::sub(
                                metadata.source_payload_offset,
                                expected_payload_offset, payload_delta))) {
        return out.reject(::rund::replay::Code::CodecInvariantInvalid);
      }
      if ((sequence_exceptional && !out.varuint(sequence_delta)) ||
          (source_changed && !out.varuint(metadata.input_source)) ||
          (schema_changed && !out.varuint(metadata.input_schema)) ||
          (event_range && (!out.varuint(event_delta) ||
                           !out.varuint(metadata.source_event_count))) ||
          (payload_range && (!out.varuint(payload_delta) ||
                             !out.varuint(metadata.source_payload_count))) ||
          (source_hash_stored && !out.fixed64(metadata.source_hash))) {
        return false;
      }
      previous_input_source = metadata.input_source;
      previous_input_schema = metadata.input_schema;
      previous_input_sequence = metadata.input_sequence;
      first_input = false;
      if (!rund::kernel::checked::add(metadata.source_event_offset,
                                      metadata.source_event_count,
                                      expected_event_offset) ||
          !rund::kernel::checked::add(metadata.source_payload_offset,
                                      metadata.source_payload_count,
                                      expected_payload_offset)) {
        return out.reject(::rund::replay::Code::CodecInvariantInvalid);
      }
    }
    if (payload_hash_stored && !out.fixed64(metadata.payload_hash.value)) {
      return false;
    }
    if (piece_count_exceptional) {
      if (!out.varuint(static_cast<std::uint64_t>(record.pieces.size()))) {
        return false;
      }
      std::uint64_t completed_bytes = 0u;
      for (const payload::ArchivePiece &piece : record.pieces) {
        if (piece.chunk_id >= archive.chunks.size()) {
          return out.reject(::rund::replay::Code::CodecInvariantInvalid);
        }
        const payload::ArchiveChunk &chunk =
            archive.chunks[static_cast<std::size_t>(piece.chunk_id)];
        if (piece.offset != 0u || piece.size != chunk.uncompressed_bytes ||
            !rund::kernel::checked::add(completed_bytes, piece.size,
                                        completed_bytes)) {
          return out.reject(::rund::replay::Code::CodecInvariantInvalid);
        }
        if (!out.varuint(chunk_delta(piece.chunk_id, previous_chunk))) {
          return false;
        }
        previous_chunk = piece.chunk_id;
      }
      if (completed_bytes != metadata.completed_bytes) {
        return out.reject(::rund::replay::Code::CodecInvariantInvalid);
      }
    } else {
      const payload::ArchivePiece &piece = record.pieces.front();
      const std::uint64_t chunk = piece.chunk_id;
      const payload::ArchiveChunk &stored =
          archive.chunks[static_cast<std::size_t>(chunk)];
      if (piece.offset != 0u || piece.size != stored.uncompressed_bytes ||
          metadata.completed_bytes != stored.uncompressed_bytes) {
        return out.reject(::rund::replay::Code::CodecInvariantInvalid);
      }
      if (chunk_changed && !out.varuint(chunk_delta(chunk, previous_chunk))) {
        return false;
      }
      previous_chunk = chunk;
    }
  }
  if (!out.varuint(
          static_cast<std::uint64_t>(archive.diagnostic.records.size())) ||
      !out.varuint(
          static_cast<std::uint64_t>(archive.diagnostic.bytes.size())) ||
      !out.fixed64(archive.diagnostic.hash) ||
      !out.varuint(archive.diagnostic.report.retained_bytes) ||
      !out.varuint(archive.diagnostic.report.retained_records) ||
      !out.varuint(archive.diagnostic.report.evicted_records) ||
      !out.varuint(archive.diagnostic.report.dropped_records)) {
    return false;
  }
  for (const payload::DiagnosticRecord &record : archive.diagnostic.records) {
    if (!out.u8(static_cast<std::uint8_t>(record.role)) ||
        !out.varuint(record.event_sequence) ||
        !out.varuint(static_cast<std::uint64_t>(record.kind)) ||
        !out.varuint(record.offset) || !out.varuint(record.byte_count) ||
        !out.fixed64(record.payload_hash.value)) {
      return false;
    }
  }
  return out.raw(archive.diagnostic.bytes.span());
}

bool read_payload(Reader &in, Admission &admission,
                  const std::span<const ::rund::host::Event> events,
                  payload::Archive &archive) {
  std::uint8_t present = 0u;
  if (!in.u8(present) || present > 1u) {
    return false;
  }
  if (present == 0u) {
    return true;
  }
  std::uint64_t chunk_count = 0u;
  std::size_t chunk_size = 0u;
  if (!in.fixed64(archive.payload_hash) || !in.varuint(chunk_count) ||
      !admission.entries(chunk_count) || !size(chunk_count, chunk_size)) {
    return false;
  }
  archive.chunks.resize(chunk_size);
  std::uint64_t encoded_total = 0u;
  std::uint64_t uncompressed_total = 0u;
  for (std::size_t index = 0u; index < chunk_size; ++index) {
    if (!read_chunk(in, admission, index, archive.chunks[index], encoded_total,
                    uncompressed_total)) {
      return false;
    }
  }

  std::uint64_t record_count = 0u;
  std::size_t record_size = 0u;
  if (!in.varuint(record_count) || !admission.entries(record_count) ||
      !size(record_count, record_size)) {
    return false;
  }
  archive.records.resize(record_size);
  std::uint64_t logical_total = 0u;
  std::uint64_t previous_host_sequence = 0u;
  std::uint64_t previous_input_source = 0u;
  std::uint64_t previous_input_schema = 0u;
  std::uint64_t previous_input_sequence = 0u;
  bool first_input = true;
  std::uint64_t expected_event_offset = 0u;
  std::uint64_t expected_payload_offset = 0u;
  std::uint64_t previous_chunk = 0u;
  std::size_t host_event_index = 0u;
  for (payload::ArchiveRecord &record : archive.records) {
    payload::Record &metadata = record.metadata;
    std::uint64_t fields_value = 0u;
    if (!in.varuint(fields_value) || fields_value > kRecordFieldMask) {
      return false;
    }
    const auto fields = static_cast<std::uint16_t>(fields_value);
    const bool input = (fields & InputRole) != 0u;
    constexpr std::uint16_t kInputOnlyFields =
        SourceChanged | SchemaChanged | EventRangePresent |
        PayloadRangePresent | SourceHashStored;
    if ((!input && (fields & kInputOnlyFields) != 0u) ||
        ((fields & SourceHashStored) != 0u) !=
            ((fields & (EventRangePresent | PayloadRangePresent)) != 0u) ||
        ((fields & PieceCountExceptional) != 0u &&
         (fields & ChunkChanged) != 0u) ||
        ((fields & PayloadHashStored) != 0u) !=
            ((fields & PieceCountExceptional) != 0u)) {
      return false;
    }
    metadata.role = input ? payload::Role::Input : payload::Role::Host;
    const bool sequence_exceptional = (fields & SequenceExceptional) != 0u;
    std::uint64_t sequence_delta = 0u;
    if (sequence_exceptional &&
        (!in.varuint(sequence_delta) || sequence_delta == 0u)) {
      return false;
    }
    if (metadata.role == payload::Role::Host) {
      if (!next_sequence(previous_host_sequence, false, sequence_delta,
                         metadata.event_sequence)) {
        return false;
      }
      while (host_event_index < events.size() &&
             events[host_event_index].sequence < metadata.event_sequence) {
        ++host_event_index;
      }
      if (host_event_index == events.size() ||
          events[host_event_index].sequence != metadata.event_sequence) {
        return false;
      }
      metadata.kind = events[host_event_index].kind;
      previous_host_sequence = metadata.event_sequence;
    } else {
      metadata.input_source = previous_input_source;
      metadata.input_schema = previous_input_schema;
      if (!next_sequence(previous_input_sequence, first_input, sequence_delta,
                         metadata.input_sequence)) {
        return false;
      }
      if (((fields & SourceChanged) != 0u &&
           (!in.varuint(metadata.input_source) ||
            metadata.input_source == previous_input_source)) ||
          ((fields & SchemaChanged) != 0u &&
           (!in.varuint(metadata.input_schema) ||
            metadata.input_schema == previous_input_schema))) {
        return false;
      }
      metadata.source_event_offset = expected_event_offset;
      metadata.source_payload_offset = expected_payload_offset;
      if ((fields & EventRangePresent) != 0u) {
        std::uint64_t offset_delta = 0u;
        if (!in.varuint(offset_delta) ||
            !in.varuint(metadata.source_event_count) ||
            (offset_delta == 0u && metadata.source_event_count == 0u)) {
          return false;
        }
        if (!rund::kernel::checked::add(metadata.source_event_offset,
                                        offset_delta,
                                        metadata.source_event_offset)) {
          return false;
        }
      }
      if ((fields & PayloadRangePresent) != 0u) {
        std::uint64_t offset_delta = 0u;
        if (!in.varuint(offset_delta) ||
            !in.varuint(metadata.source_payload_count) ||
            (offset_delta == 0u && metadata.source_payload_count == 0u)) {
          return false;
        }
        if (!rund::kernel::checked::add(metadata.source_payload_offset,
                                        offset_delta,
                                        metadata.source_payload_offset)) {
          return false;
        }
      }
      if ((fields & SourceHashStored) != 0u) {
        if (!in.fixed64(metadata.source_hash)) {
          return false;
        }
      } else {
        metadata.source_hash =
            payload::SourceRangeHasher(metadata.source_event_offset, 0u,
                                       metadata.source_payload_offset, 0u)
                .Finish();
      }
      previous_input_source = metadata.input_source;
      previous_input_schema = metadata.input_schema;
      previous_input_sequence = metadata.input_sequence;
      first_input = false;
      if (!rund::kernel::checked::add(metadata.source_event_offset,
                                      metadata.source_event_count,
                                      expected_event_offset) ||
          !rund::kernel::checked::add(metadata.source_payload_offset,
                                      metadata.source_payload_count,
                                      expected_payload_offset)) {
        return false;
      }
    }

    const bool payload_hash_stored = (fields & PayloadHashStored) != 0u;
    if (payload_hash_stored && !in.fixed64(metadata.payload_hash.value)) {
      return false;
    }
    const bool exceptional_pieces = (fields & PieceCountExceptional) != 0u;
    std::uint64_t piece_count = exceptional_pieces ? 0u : 1u;
    if (exceptional_pieces && (!in.varuint(piece_count) || piece_count == 1u)) {
      return false;
    }
    std::size_t piece_size = 0u;
    if (!admission.entries(piece_count) || !size(piece_count, piece_size)) {
      return false;
    }
    record.pieces.resize(piece_size);
    for (payload::ArchivePiece &piece : record.pieces) {
      std::uint64_t chunk_delta = 0u;
      const bool changed = exceptional_pieces || (fields & ChunkChanged) != 0u;
      if ((changed && !in.varuint(chunk_delta)) ||
          (!exceptional_pieces && changed && chunk_delta == 0u)) {
        return false;
      }
      piece.chunk_id = chunk_id(previous_chunk, chunk_delta);
      if (piece.chunk_id >= archive.chunks.size()) {
        return false;
      }
      const std::uint64_t bytes =
          archive.chunks[static_cast<std::size_t>(piece.chunk_id)]
              .uncompressed_bytes;
      if (bytes > std::numeric_limits<std::uint32_t>::max()) {
        return false;
      }
      piece.offset = 0u;
      piece.size = static_cast<std::uint32_t>(bytes);
      if (!rund::kernel::checked::add(metadata.completed_bytes, bytes,
                                      metadata.completed_bytes)) {
        return false;
      }
      previous_chunk = piece.chunk_id;
    }
    if (!admission.payload(logical_total, metadata.completed_bytes)) {
      return false;
    }
    if (!payload_hash_stored) {
      if (record.pieces.size() == 1u) {
        metadata.payload_hash = archive
                                    .chunks[static_cast<std::size_t>(
                                        record.pieces.front().chunk_id)]
                                    .uncompressed_hash;
      } else if (record.pieces.empty()) {
        metadata.payload_hash.value = payload::ByteHash{}.Finish();
      } else {
        return false;
      }
    }
  }

  std::uint64_t diagnostic_count = 0u;
  std::uint64_t diagnostic_bytes = 0u;
  std::size_t diagnostic_size = 0u;
  std::size_t diagnostic_byte_size = 0u;
  if (!in.varuint(diagnostic_count) || !admission.entries(diagnostic_count) ||
      !size(diagnostic_count, diagnostic_size) ||
      !in.varuint(diagnostic_bytes) ||
      diagnostic_bytes > admission.limits().max_payload_bytes ||
      !size(diagnostic_bytes, diagnostic_byte_size) ||
      !in.fixed64(archive.diagnostic.hash) ||
      !in.varuint(archive.diagnostic.report.retained_bytes) ||
      !in.varuint(archive.diagnostic.report.retained_records) ||
      !in.varuint(archive.diagnostic.report.evicted_records) ||
      !in.varuint(archive.diagnostic.report.dropped_records)) {
    return false;
  }
  archive.diagnostic.records.resize(diagnostic_size);
  for (payload::DiagnosticRecord &record : archive.diagnostic.records) {
    std::uint8_t role = 0u;
    std::uint64_t kind = 0u;
    if (!in.u8(role) ||
        role > static_cast<std::uint8_t>(
                   payload::DiagnosticRole::NetworkIngress) ||
        !in.varuint(record.event_sequence) || !in.varuint(kind) ||
        kind > std::numeric_limits<std::uint16_t>::max() ||
        !in.varuint(record.offset) || !in.varuint(record.byte_count) ||
        !in.fixed64(record.payload_hash.value)) {
      return false;
    }
    record.role = static_cast<payload::DiagnosticRole>(role);
    record.kind = static_cast<::rund::host::EventKind>(kind);
  }
  std::span<const std::byte> diagnostic{};
  if (!in.take(diagnostic_byte_size, diagnostic)) {
    return false;
  }
  std::byte *diagnostic_data = nullptr;
  archive.diagnostic.bytes =
      payload::Bytes::create(diagnostic_byte_size, diagnostic_data);
  if (diagnostic_byte_size != 0u) {
    std::memcpy(diagnostic_data, diagnostic.data(), diagnostic_byte_size);
  }

  std::size_t compact_size = 0u;
  if (!size(encoded_total, compact_size)) {
    return false;
  }
  std::byte *compact_data = nullptr;
  const payload::Bytes compact =
      payload::Bytes::create(compact_size, compact_data);
  std::size_t compact_offset = 0u;
  for (payload::ArchiveChunk &chunk : archive.chunks) {
    std::size_t encoded_size = 0u;
    std::size_t encoded_offset = 0u;
    std::span<const std::byte> encoded{};
    if (!size(chunk.encoded_bytes, encoded_size) ||
        !size(chunk.segment_offset, encoded_offset) ||
        !in.view(encoded_offset, encoded_size, encoded)) {
      return false;
    }
    if (!encoded.empty()) {
      std::memcpy(compact_data + compact_offset, encoded.data(),
                  encoded.size());
    }
    chunk.encoded = compact.slice(compact_offset, encoded.size());
    chunk.segment_offset = 0u;
    compact_offset += encoded.size();
  }
  if (compact_offset != compact_size) {
    return false;
  }

  archive.storage.mode = ::rund::replay::StorageMode::Memory;
  archive.storage.logical_bytes = logical_total;
  archive.storage.encoded_bytes = encoded_total;
  archive.storage.retained_bytes = encoded_total;
  archive.storage.chunk_count = chunk_count;
  return true;
}

} // namespace rund::node::replay_detail::artifact
