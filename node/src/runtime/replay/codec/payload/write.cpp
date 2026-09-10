#include "../local.hpp"

#include "../../host/payload/backend.hpp"
#include "../../host/payload/chunk.hpp"
#include "../../host/payload/store.hpp"
#include "internal.hpp"

#include <node/runtime/replay/host/archive.hpp>

#include <cstddef>
#include <cstdint>
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

// Schema 1 chunk deltas inhabit Z/(2^64). Every decoded representative is
// admitted against archive.chunks.size(). Keeping the modular projection
// explicit preserves the established bytes for both increasing and decreasing
// deduplicated chunk references without introducing a second grammar.
[[nodiscard]] constexpr std::uint64_t
chunk_delta(const std::uint64_t chunk, const std::uint64_t previous) noexcept {
  return chunk - previous;
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
    if (!payload_codec::next_sequence(
            input ? previous_input_sequence : previous_host_sequence,
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

} // namespace rund::node::replay_detail::artifact
