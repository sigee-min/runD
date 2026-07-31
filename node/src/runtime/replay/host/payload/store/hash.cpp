#include "../hash.hpp"
#include "../backend.hpp"
#include "../store.hpp"

#include <optional>

namespace rund::node::replay_detail::payload {
namespace {

[[nodiscard]] std::optional<std::uint64_t>
ComputeRecordHash(const StoredRecord &record,
                  const std::span<const Piece> pieces, const Backend &backend) {
  RecordHash hash{record.metadata, record.metadata.completed_bytes,
                  static_cast<std::uint64_t>(pieces.size())};
  std::uint64_t byte_count = 0u;
  for (const Piece piece : pieces) {
    if (piece.blob_index >= backend.blobs().size()) {
      return std::nullopt;
    }
    const Blob &blob = backend.blobs()[piece.blob_index];
    if (byte_count > record.metadata.completed_bytes ||
        blob.uncompressed_bytes >
            record.metadata.completed_bytes - byte_count) {
      return std::nullopt;
    }
    byte_count += blob.uncompressed_bytes;
    hash.Append(blob.payload_hash, blob.uncompressed_bytes);
  }
  if (byte_count != record.metadata.completed_bytes) {
    return std::nullopt;
  }
  return hash.Finish();
}

} // namespace

std::optional<std::uint64_t>
Store::SourceRangeHash(const std::uint64_t event_offset,
                       const std::span<const ::rund::host::Event> events,
                       const std::uint64_t payload_offset,
                       const std::uint64_t payload_count) const {
  if (payload_offset > host_record_indices_.size() ||
      payload_count > host_record_indices_.size() - payload_offset) {
    return std::nullopt;
  }
  std::size_t cursor = static_cast<std::size_t>(payload_offset);
  const auto next_payload =
      [this, &cursor]() noexcept -> std::optional<SourcePayloadBinding> {
    if (cursor >= host_record_indices_.size()) {
      return std::nullopt;
    }
    const std::size_t record_index = host_record_indices_[cursor++];
    if (record_index >= records_.size()) {
      return std::nullopt;
    }
    const StoredRecord &record = records_[record_index];
    return SourcePayloadBinding{
        .event_sequence = record.metadata.event_sequence,
        .kind = record.metadata.kind,
        .completed_bytes = record.metadata.completed_bytes,
        .payload_hash = record.metadata.payload_hash};
  };
  return ComputeSourceRangeHash(event_offset, events, payload_offset,
                                payload_count, next_payload);
}

bool Store::ComputeRecordHashes() {
  for (StoredRecord &record : records_) {
    const std::optional<std::uint64_t> hash =
        ComputeRecordHash(record, pieces(record), *backend_);
    if (!hash.has_value()) {
      return false;
    }
    record.record_hash = *hash;
  }
  return true;
}

std::uint64_t Store::payload_hash() const {
  if (records_.empty()) {
    return 0u;
  }
  if (loaded_archive_) {
    return loaded_payload_hash_;
  }
  ArchiveHash hash{static_cast<std::uint64_t>(records_.size())};
  for (const StoredRecord &record : records_) {
    hash.Append(record.record_hash);
  }
  return hash.Finish();
}

} // namespace rund::node::replay_detail::payload
