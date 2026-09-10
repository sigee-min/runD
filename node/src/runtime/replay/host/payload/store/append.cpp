#include "local.hpp"

#include "../backend.hpp"
#include "../hash.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <limits>
#include <utility>

namespace rund::node::replay_detail::payload {

bool Store::AppendRecord(StoredRecord record, const Capture payload,
                         ::rund::node::replay_detail::payload::Bytes adopted) {
  if (!payload) {
    return false;
  }
  const std::span<const std::byte> bytes = payload.bytes();
  if (record.metadata.completed_bytes != bytes.size() ||
      record.metadata.payload_hash.value != payload.hash().value ||
      (!adopted.empty() &&
       (adopted.data() != bytes.data() || adopted.size() != bytes.size()))) {
    return false;
  }
  if (loaded_archive_ && !ComputeRecordHashes()) {
    return false;
  }
  if ((record.metadata.role ==
           ::rund::node::replay_detail::payload::Role::Host &&
       host_record_indices_.size() >= limits_.hosts) ||
      (record.metadata.role ==
           ::rund::node::replay_detail::payload::Role::Input &&
       input_record_indices_.size() >= limits_.inputs)) {
    return false;
  }
  loaded_archive_ = false;
  loaded_payload_hash_ = 0u;
  const std::uint64_t logical_added = static_cast<std::uint64_t>(bytes.size());
  if (!store_detail::fits_budget(logical_bytes_, logical_added,
                                 limits_.bytes)) {
    return false;
  }
  const std::size_t chunk_count =
      bytes.empty() ? 0u : 1u + (bytes.size() - 1u) / kChunkBytes;
  if (chunk_count > piece_scratch_.capacity() ||
      chunk_count > pieces_.capacity() - pieces_.size() ||
      backend_->blobs().size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  piece_scratch_.clear();
  staged_blobs_.clear();
  staged_index_.clear();
  RecordHash record_hasher{record.metadata,
                           static_cast<std::uint64_t>(bytes.size()),
                           static_cast<std::uint64_t>(chunk_count)};
  const auto make_blob = [&adopted](const std::size_t offset,
                                    const std::span<const std::byte> chunk,
                                    const ::rund::StableHash chunk_hash) {
    return adopted.empty()
               ? Encode(chunk, chunk_hash)
               : Blob{.payload_hash = chunk_hash,
                      .uncompressed_bytes = chunk.size(),
                      .encoded_bytes = chunk.size(),
                      .codec = Codec::Raw,
                      .encoded = adopted.slice(offset, chunk.size())};
  };
  std::uint64_t staged_encoded_bytes = 0u;
  const std::size_t base_blob = backend_->blobs().size();
  for (std::size_t offset = 0u; offset < bytes.size(); offset += kChunkBytes) {
    const std::size_t size = std::min(kChunkBytes, bytes.size() - offset);
    const std::span<const std::byte> chunk = bytes.subspan(offset, size);
    const ::rund::StableHash chunk_hash =
        chunk_count == 1u
            ? record.metadata.payload_hash
            : ::rund::host::hash_bytes(chunk.data(), chunk.size());
    record_hasher.Append(chunk_hash, static_cast<std::uint64_t>(size));
    if (const auto existing = backend_->Find(chunk, chunk_hash);
        existing.has_value()) {
      piece_scratch_.push_back(Piece{.blob_index = existing.value()});
      continue;
    }
    if (const auto staged = staged_index_.find(
            chunk_hash.value,
            [&](const std::uint32_t index) {
              return index < staged_blobs_.size() &&
                     ::rund::node::replay_detail::payload::Matches(
                         staged_blobs_[index], chunk, chunk_hash);
            });
        staged.has_value()) {
      piece_scratch_.push_back(
          Piece{.blob_index = static_cast<std::uint32_t>(base_blob + *staged)});
      continue;
    }
    if (staged_blobs_.size() >= staged_blobs_.capacity() ||
        base_blob + staged_blobs_.size() >
            std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    Blob blob = make_blob(offset, chunk, chunk_hash);
    if (!rund::kernel::checked::add(staged_encoded_bytes, blob.encoded_bytes,
                                    staged_encoded_bytes)) {
      return false;
    }
    piece_scratch_.push_back(Piece{.blob_index = static_cast<std::uint32_t>(
                                       base_blob + staged_blobs_.size())});
    const std::uint32_t staged =
        static_cast<std::uint32_t>(staged_blobs_.size());
    staged_blobs_.push_back(std::move(blob));
    if (!staged_index_.insert(chunk_hash.value, staged)) {
      return false;
    }
  }
  record.record_hash = record_hasher.Finish();
  if (!store_detail::fits_budget(backend_->encoded_bytes(),
                                 staged_encoded_bytes, limits_.bytes)) {
    return false;
  }
  if (!backend_->CanAppend(staged_blobs_)) {
    return false;
  }
  const BatchResult appended = backend_->Append(staged_blobs_);
  if (!appended.ok() || appended.first_blob != base_blob ||
      appended.blob_count != staged_blobs_.size()) {
    return false;
  }
  if (appended.blob_count != 0u) {
    publication_ = {};
    publication_chunks_ = 0u;
  }
  const std::size_t record_capacity = records_.capacity();
  const std::size_t host_capacity = host_record_indices_.capacity();
  const std::size_t input_capacity = input_record_indices_.capacity();
  const std::size_t piece_capacity = pieces_.capacity();
  record.piece_offset = static_cast<std::uint32_t>(pieces_.size());
  record.piece_count = static_cast<std::uint32_t>(piece_scratch_.size());
  pieces_.insert(pieces_.end(), piece_scratch_.begin(), piece_scratch_.end());
  Commit(std::move(record), logical_added);
  const std::uint64_t changes =
      static_cast<std::uint64_t>(records_.capacity() != record_capacity) +
      static_cast<std::uint64_t>(host_record_indices_.capacity() !=
                                 host_capacity) +
      static_cast<std::uint64_t>(input_record_indices_.capacity() !=
                                 input_capacity) +
      static_cast<std::uint64_t>(pieces_.capacity() != piece_capacity);
  growths_ = ::rund::detail::counter::SaturatingAdd(growths_, changes);
  return true;
}

} // namespace rund::node::replay_detail::payload
