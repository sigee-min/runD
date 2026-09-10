#include "../local.hpp"

#include "../../host/payload/backend.hpp"
#include "../../host/payload/chunk.hpp"
#include "../../host/payload/hash.hpp"
#include "internal.hpp"

#include <node/runtime/replay/host/archive.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>

namespace rund::node::replay_detail::artifact {
namespace {

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

[[nodiscard]] constexpr std::uint64_t
chunk_id(const std::uint64_t previous, const std::uint64_t delta) noexcept {
  return previous + delta;
}

} // namespace

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
      if (!payload_codec::next_sequence(previous_host_sequence, false,
                                        sequence_delta,
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
      if (!payload_codec::next_sequence(previous_input_sequence, first_input,
                                        sequence_delta,
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
