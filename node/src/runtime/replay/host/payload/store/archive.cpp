#include "../backend.hpp"
#include "../store.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rund::node::replay_detail::payload {

Store::Publication Store::Compact(
    std::vector<::rund::node::replay_detail::payload::ArchiveChunk> &chunks) {
  if (chunks.empty()) {
    return {};
  }

  std::size_t total = 0u;
  for (const ::rund::node::replay_detail::payload::ArchiveChunk &chunk :
       chunks) {
    if (chunk.encoded.size() >
        std::numeric_limits<std::size_t>::max() - total) {
      throw std::length_error{"replay payload publication exceeds size_t"};
    }
    total += chunk.encoded.size();
  }
  if (total > std::numeric_limits<std::uint64_t>::max()) {
    throw std::length_error{"replay payload publication exceeds uint64"};
  }
  if (total == 0u) {
    return {};
  }

  const auto first_chunk = std::find_if(
      chunks.begin(), chunks.end(),
      [](const ::rund::node::replay_detail::payload::ArchiveChunk &chunk) {
        return !chunk.encoded.empty();
      });
  if (first_chunk == chunks.end()) {
    throw std::length_error{"replay payload publication size mismatch"};
  }
  const ::rund::node::replay_detail::payload::Bytes &first =
      first_chunk->encoded;
  bool exact = first.owner_ != nullptr && first.owner_data_ != nullptr &&
               first.data_ == first.owner_data_ && first.owner_size_ == total &&
               first.retained_bytes_ == total;
  std::size_t offset = 0u;
  for (const ::rund::node::replay_detail::payload::ArchiveChunk &chunk :
       chunks) {
    const ::rund::node::replay_detail::payload::Bytes &bytes = chunk.encoded;
    if (bytes.empty()) {
      continue;
    }
    const bool same_owner = !first.owner_.owner_before(bytes.owner_) &&
                            !bytes.owner_.owner_before(first.owner_);
    if (!exact || !same_owner || bytes.data_ != first.owner_data_ + offset) {
      exact = false;
    }
    offset += bytes.size();
  }
  if (exact) {
    return Publication{.retained_bytes = static_cast<std::uint64_t>(total)};
  }

  std::byte *output = nullptr;
  const ::rund::node::replay_detail::payload::Bytes compact =
      ::rund::node::replay_detail::payload::Bytes::create(total, output);
  offset = 0u;
  for (const ::rund::node::replay_detail::payload::ArchiveChunk &chunk :
       chunks) {
    if (!chunk.encoded.empty()) {
      std::memcpy(output + offset, chunk.encoded.data(), chunk.encoded.size());
      offset += chunk.encoded.size();
    }
  }
  offset = 0u;
  for (::rund::node::replay_detail::payload::ArchiveChunk &chunk : chunks) {
    const std::size_t size = chunk.encoded.size();
    chunk.encoded = compact.slice(offset, size);
    offset += size;
  }
  return Publication{.retained_bytes = static_cast<std::uint64_t>(total),
                     .copied_bytes = static_cast<std::uint64_t>(total)};
}

::rund::node::replay_detail::payload::Archive Store::Archive() const {
  ::rund::node::replay_detail::payload::Archive archive{};
  archive.records.reserve(records_.size());
  archive.chunks.reserve(backend_->blobs().size());
  archive.payload_hash = payload_hash();
  archive.storage = backend_->Report();
  archive.spill_generation =
      backend_->blobs().empty() ? nullptr : backend_->generation();
  archive.storage.logical_bytes = logical_bytes_;
  archive.storage.growths =
      ::rund::detail::counter::SaturatingAdd(archive.storage.growths, growths_);
  archive.diagnostic =
      loaded_diagnostic_active_ ? loaded_diagnostic_ : diagnostic_.Archive();
  const auto &blobs = backend_->blobs();
  const auto &refs = backend_->refs();
  const bool cached = storage_.mode == ::rund::replay::StorageMode::Memory &&
                      publication_chunks_ == blobs.size() &&
                      publication_.size() == backend_->encoded_bytes();
  std::size_t publication_offset = 0u;
  for (std::size_t index = 0u; index < blobs.size(); ++index) {
    const Blob &blob = blobs[index];
    ::rund::node::replay_detail::payload::ArchiveChunk chunk{
        .chunk_id = static_cast<std::uint64_t>(index),
        .codec = blob.codec,
        .uncompressed_bytes = blob.uncompressed_bytes,
        .encoded_bytes = blob.encoded_bytes,
        .uncompressed_hash = blob.payload_hash,
        .encoded =
            cached ? publication_.slice(publication_offset, blob.encoded.size())
                   : blob.encoded,
        .spilled = storage_.mode == ::rund::replay::StorageMode::Spill,
    };
    if (chunk.spilled && index < refs.size()) {
      chunk.segment_index = refs[index].segment_index;
      chunk.segment_offset = refs[index].segment_offset;
      chunk.segment_record_bytes = refs[index].record_bytes;
    }
    publication_offset += blob.encoded.size();
    archive.chunks.push_back(std::move(chunk));
  }
  if (storage_.mode == ::rund::replay::StorageMode::Memory) {
    const Publication publication = Compact(archive.chunks);
    archive.storage.retained_bytes = publication.retained_bytes;
    archive.storage.copied_bytes = publication.copied_bytes;
    if (!archive.chunks.empty() && publication.retained_bytes != 0u) {
      const auto first = std::find_if(
          archive.chunks.begin(), archive.chunks.end(),
          [](const ::rund::node::replay_detail::payload::ArchiveChunk &chunk) {
            return !chunk.encoded.empty();
          });
      if (first != archive.chunks.end()) {
        const ::rund::node::replay_detail::payload::Bytes &bytes =
            first->encoded;
        publication_ = ::rund::node::replay_detail::payload::Bytes{
            bytes.owner_,      bytes.owner_data_, bytes.owner_data_,
            bytes.owner_size_, bytes.owner_size_, bytes.retained_bytes_};
        publication_chunks_ = archive.chunks.size();
      }
    }
  }
  for (const StoredRecord &stored : records_) {
    const std::span<const Piece> stored_pieces = pieces(stored);
    ::rund::node::replay_detail::payload::ArchiveRecord ref{
        .metadata = stored.metadata,
    };
    ref.pieces.reserve(stored_pieces.size());
    for (const Piece piece : stored_pieces) {
      if (piece.blob_index < blobs.size()) {
        ref.pieces.push_back(::rund::node::replay_detail::payload::ArchivePiece{
            .chunk_id = piece.blob_index,
            .offset = 0u,
            .size = static_cast<std::uint32_t>(
                blobs[piece.blob_index].uncompressed_bytes),
        });
      }
    }
    archive.records.push_back(std::move(ref));
  }
  return archive;
}

BuildResult Build(::rund::node::replay_detail::payload::Archive archive,
                  ::rund::replay::Storage storage,
                  ::rund::replay::Diagnostic diagnostic) {
  const std::optional<Limits> limits = Limits::archive(archive);
  if (!limits.has_value() || limits->bytes > storage.max_bytes) {
    return BuildResult{.code =
                           ::rund::replay::Code::HostPayloadCapacityExceeded,
                       .store = Store{}};
  }
  Store store{std::move(storage), *limits, diagnostic};
  if (!store.LoadArchive(std::move(archive))) {
    return BuildResult{.code = ::rund::replay::Code::HostPayloadHashInvalid,
                       .store = Store{}};
  }
  return BuildResult{.code = ::rund::replay::Code::Ok,
                     .store = std::move(store)};
}

} // namespace rund::node::replay_detail::payload
