#include "local.hpp"

#include <utility>

namespace rund::node::replay_detail::payload {

RecordLoad
LoadRecords(const ::rund::node::replay_detail::payload::Archive &archive) {
  RecordLoad result{};
  result.records.reserve(archive.records.size());
  std::size_t piece_count = 0u;
  for (const ::rund::node::replay_detail::payload::ArchiveRecord &record :
       archive.records) {
    if (record.pieces.size() >
        std::numeric_limits<std::size_t>::max() - piece_count) {
      return RecordLoad{.code =
                            ::rund::replay::Code::HostPayloadCapacityExceeded};
    }
    piece_count += record.pieces.size();
  }
  if (piece_count > std::numeric_limits<std::uint32_t>::max()) {
    return RecordLoad{.code =
                          ::rund::replay::Code::HostPayloadCapacityExceeded};
  }
  result.pieces.reserve(piece_count);
  for (const ::rund::node::replay_detail::payload::ArchiveRecord &ref :
       archive.records) {
    StoredRecord record{
        .metadata = ref.metadata,
        .piece_offset = static_cast<std::uint32_t>(result.pieces.size()),
        .piece_count = static_cast<std::uint32_t>(ref.pieces.size())};
    const bool host_record =
        ref.metadata.role == ::rund::node::replay_detail::payload::Role::Host;
    const bool input_record =
        ref.metadata.role == ::rund::node::replay_detail::payload::Role::Input;
    if ((!host_record && !input_record) ||
        (host_record &&
         (ref.metadata.input_source != 0u || ref.metadata.input_schema != 0u ||
          ref.metadata.input_sequence != 0u ||
          ref.metadata.source_event_offset != 0u ||
          ref.metadata.source_event_count != 0u ||
          ref.metadata.source_payload_offset != 0u ||
          ref.metadata.source_payload_count != 0u ||
          ref.metadata.source_hash != 0u)) ||
        (input_record &&
         (ref.metadata.event_sequence != 0u ||
          ref.metadata.kind != ::rund::host::EventKind::None ||
          ref.metadata.input_source == 0u || ref.metadata.input_schema == 0u ||
          ref.metadata.source_hash == 0u ||
          !rund::kernel::checked::add(ref.metadata.source_event_offset,
                                      ref.metadata.source_event_count) ||
          !rund::kernel::checked::add(ref.metadata.source_payload_offset,
                                      ref.metadata.source_payload_count)))) {
      return RecordLoad{.code = ::rund::replay::Code::HostBadValue};
    }
    std::uint64_t piece_bytes = 0u;
    for (const ::rund::node::replay_detail::payload::ArchivePiece &piece :
         ref.pieces) {
      if (piece.chunk_id >= archive.chunks.size() || piece.offset != 0u ||
          piece.size != archive.chunks[static_cast<std::size_t>(piece.chunk_id)]
                            .uncompressed_bytes) {
        return RecordLoad{.code = ::rund::replay::Code::HostBadIndex};
      }
      if (!rund::kernel::checked::add(piece_bytes, piece.size, piece_bytes)) {
        return RecordLoad{
            .code = ::rund::replay::Code::HostPayloadCapacityExceeded};
      }
      result.pieces.push_back(
          Piece{.blob_index = static_cast<std::uint32_t>(piece.chunk_id)});
    }
    if (piece_bytes != ref.metadata.completed_bytes ||
        !rund::kernel::checked::add(result.logical_bytes,
                                    ref.metadata.completed_bytes,
                                    result.logical_bytes)) {
      return RecordLoad{.code =
                            ::rund::replay::Code::HostPayloadCapacityExceeded};
    }
    result.records.push_back(std::move(record));
  }
  result.code = ::rund::replay::Code::Ok;
  return result;
}

} // namespace rund::node::replay_detail::payload
