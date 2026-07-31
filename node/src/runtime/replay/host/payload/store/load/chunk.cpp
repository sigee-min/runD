#include "local.hpp"

#include <utility>

namespace rund::node::replay_detail::payload {

ChunkLoad LoadChunks(::rund::node::replay_detail::payload::Archive &archive,
                     const ::rund::replay::Storage &storage) {
  ChunkLoad result{};
  result.blobs.reserve(archive.chunks.size());
  result.refs.reserve(archive.chunks.size());
  for (std::size_t index = 0u; index < archive.chunks.size(); ++index) {
    ::rund::node::replay_detail::payload::ArchiveChunk &chunk =
        archive.chunks[index];
    if (chunk.chunk_id != index) {
      return ChunkLoad{.code = ::rund::replay::Code::HostBadIndex};
    }
    if (chunk.encoded_bytes !=
            static_cast<std::uint64_t>(chunk.encoded.size()) &&
        !chunk.spilled) {
      return ChunkLoad{.code = ::rund::replay::Code::HostBadValue};
    }
    Blob blob{
        .payload_hash = chunk.uncompressed_hash,
        .uncompressed_bytes = chunk.uncompressed_bytes,
        .encoded_bytes = chunk.encoded_bytes,
        .codec = chunk.codec,
        .encoded = std::move(chunk.encoded),
    };
    if (storage.mode == ::rund::replay::StorageMode::Memory && chunk.spilled) {
      return ChunkLoad{.code = ::rund::replay::Code::HostBadValue};
    }
    if (storage.mode == ::rund::replay::StorageMode::Spill && !chunk.spilled) {
      return ChunkLoad{.code = ::rund::replay::Code::HostBadValue};
    }
    if (storage.mode == ::rund::replay::StorageMode::Spill) {
      if (chunk.segment_record_bytes == 0u ||
          chunk.segment_record_bytes > storage.segment_bytes ||
          !rund::kernel::checked::add(chunk.segment_offset,
                                      chunk.segment_record_bytes)) {
        return ChunkLoad{.code = ::rund::replay::Code::HostBadValue};
      }
      result.refs.push_back(SpillRef{
          .segment_index = chunk.segment_index,
          .segment_offset = chunk.segment_offset,
          .record_bytes = chunk.segment_record_bytes,
      });
      blob.encoded = {};
    }
    if (!rund::kernel::checked::add(result.encoded_bytes, chunk.encoded_bytes,
                     result.encoded_bytes)) {
      return ChunkLoad{.code =
                           ::rund::replay::Code::HostPayloadCapacityExceeded};
    }
    result.blobs.push_back(std::move(blob));
  }
  result.code = ::rund::replay::Code::Ok;
  return result;
}

} // namespace rund::node::replay_detail::payload
