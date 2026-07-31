#include "../backend.hpp"
#include "../hash.hpp"
#include "../store.hpp"

#include <limits>
#include <optional>
#include <utility>

namespace rund::node::replay_detail::payload {
namespace {

constexpr ::rund::replay::Code kBindingMismatch =
    ::rund::replay::Code::HostPayloadMismatch;

[[nodiscard]] bool Bound(const StoredRecord &record,
                         const Binding &binding) noexcept {
  const Record &metadata = record.metadata;
  return metadata.role == ::rund::node::replay_detail::payload::Role::Host &&
         metadata.event_sequence == binding.event_sequence &&
         metadata.kind == binding.kind &&
         metadata.completed_bytes == binding.completed_bytes &&
         metadata.payload_hash.value == binding.payload_hash.value;
}

[[nodiscard]] bool Bound(const StoredRecord &record,
                         const InputBinding &binding) noexcept {
  const Record &metadata = record.metadata;
  return metadata.role == ::rund::node::replay_detail::payload::Role::Input &&
         metadata.input_source == binding.source &&
         metadata.input_schema == binding.schema &&
         metadata.input_sequence == binding.sequence;
}

[[nodiscard]] bool WholeChunk(const StoredRecord &record,
                              const std::span<const Piece> pieces,
                              const Backend &backend) noexcept {
  if (pieces.size() != 1u) {
    return false;
  }
  const Piece piece = pieces.front();
  if (piece.blob_index >= backend.blobs().size()) {
    return false;
  }
  const Blob &blob = backend.blobs()[piece.blob_index];
  return blob.uncompressed_bytes == record.metadata.completed_bytes &&
         blob.payload_hash.value == record.metadata.payload_hash.value;
}

} // namespace

ResolveResult Store::Resolve(const std::size_t record_index) const {
  if (record_index >= records_.size()) {
    return ResolveResult{.code = ::rund::replay::Code::HostPayloadMissing,
                         .bytes = {}};
  }
  const StoredRecord &record = records_[record_index];
  if (record.metadata.completed_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ResolveResult{.code = ::rund::replay::Code::HostPayloadHashInvalid,
                         .bytes = {}};
  }
  std::vector<std::byte> bytes(
      static_cast<std::size_t>(record.metadata.completed_bytes));
  const MatchResult read = Read(record_index, bytes);
  if (!read.ok()) {
    return ResolveResult{.code = read.code, .bytes = {}};
  }
  return ResolveResult{.code = ::rund::replay::Code::Ok,
                       .bytes =
                           ::rund::node::replay_detail::payload::Bytes::freeze(
                               std::move(bytes))};
}

EncodedResult Store::Encoded(const std::size_t chunk_index) const noexcept {
  if (chunk_index > std::numeric_limits<std::uint32_t>::max()) {
    return EncodedResult{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  return backend_->Encoded(static_cast<std::uint32_t>(chunk_index));
}

MatchResult Store::Read(const std::size_t record_index,
                        const std::span<std::byte> output) const {
  if (record_index >= records_.size()) {
    return MatchResult{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  const StoredRecord &record = records_[record_index];
  if (record.metadata.completed_bytes != output.size()) {
    return MatchResult{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  const std::span<const Piece> record_pieces = pieces(record);
  std::optional<ByteHash> record_hash{};
  if (!WholeChunk(record, record_pieces, *backend_)) {
    record_hash.emplace();
  }
  std::size_t offset = 0u;
  for (const Piece piece : record_pieces) {
    if (piece.blob_index >= backend_->blobs().size()) {
      return MatchResult{.code = ::rund::replay::Code::HostPayloadMissing};
    }
    const std::uint64_t piece_bytes =
        backend_->blobs()[piece.blob_index].uncompressed_bytes;
    if (offset > output.size() ||
        piece_bytes > static_cast<std::uint64_t>(output.size() - offset)) {
      return MatchResult{.code = ::rund::replay::Code::HostPayloadHashInvalid};
    }
    const std::size_t size = static_cast<std::size_t>(piece_bytes);
    const ReadStatus read =
        backend_->ReadInto(piece.blob_index, output.subspan(offset, size),
                           record_hash ? &*record_hash : nullptr);
    if (!read.ok()) {
      return MatchResult{.code = read.code};
    }
    offset += size;
  }
  if (offset != output.size() ||
      (record_hash &&
       record_hash->Finish() != record.metadata.payload_hash.value)) {
    return MatchResult{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  return MatchResult{.code = ::rund::replay::Code::Ok};
}

MatchResult Store::ReadInput(const std::size_t input_index,
                             const InputBinding &binding,
                             const std::span<std::byte> output) const {
  const MatchResult checked = CheckInput(input_index, binding);
  if (!checked.ok()) {
    return checked;
  }
  const std::size_t record_index = input_record_indices_[input_index];
  MatchResult read = Read(record_index, output);
  if (!read.ok()) {
    read.code = ::rund::replay::Code::InputCorrupt;
  }
  return read;
}

MatchResult Store::CheckInput(const std::size_t input_index,
                              const InputBinding &binding) const {
  if (binding.source == 0u) {
    return MatchResult{.code = ::rund::replay::Code::InputIdInvalid};
  }
  if (binding.schema == 0u) {
    return MatchResult{.code = ::rund::replay::Code::InputSchemaInvalid};
  }
  if (input_index >= input_record_indices_.size()) {
    return MatchResult{.code = ::rund::replay::Code::InputMissing};
  }
  const std::size_t record_index = input_record_indices_[input_index];
  if (record_index >= records_.size() ||
      !Bound(records_[record_index], binding)) {
    return MatchResult{.code = ::rund::replay::Code::InputOrderMismatch};
  }
  return MatchResult{.code = ::rund::replay::Code::Ok};
}

MatchResult Store::ReadInto(const std::size_t record_index,
                            const Binding &binding,
                            const std::span<std::byte> output) const {
  if (record_index >= records_.size() ||
      !Bound(records_[record_index], binding)) {
    return MatchResult{.code = kBindingMismatch};
  }
  return Read(record_index, output);
}

MatchResult Store::Matches(const std::size_t record_index,
                           const Binding &binding,
                           const std::span<const std::byte> expected) const {
  if (record_index >= records_.size() ||
      !Bound(records_[record_index], binding)) {
    return MatchResult{.code = kBindingMismatch};
  }
  const StoredRecord &record = records_[record_index];
  if (record.metadata.completed_bytes != expected.size()) {
    return MatchResult{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  const std::span<const Piece> record_pieces = pieces(record);
  std::optional<ByteHash> record_hash{};
  if (!WholeChunk(record, record_pieces, *backend_)) {
    record_hash.emplace();
  }
  std::size_t offset = 0u;
  for (const Piece piece : record_pieces) {
    if (piece.blob_index >= backend_->blobs().size()) {
      return MatchResult{.code = ::rund::replay::Code::HostPayloadMissing};
    }
    const std::uint64_t chunk_bytes =
        backend_->blobs()[piece.blob_index].uncompressed_bytes;
    if (offset > expected.size() ||
        chunk_bytes > static_cast<std::uint64_t>(expected.size() - offset)) {
      return MatchResult{.code = ::rund::replay::Code::HostPayloadHashInvalid};
    }
    const std::size_t size = static_cast<std::size_t>(chunk_bytes);
    const ReadStatus matched =
        backend_->Matches(piece.blob_index, expected.subspan(offset, size),
                          record_hash ? &*record_hash : nullptr);
    if (!matched.ok()) {
      return MatchResult{.code = matched.code};
    }
    offset += size;
  }
  if (offset != expected.size() ||
      (record_hash &&
       record_hash->Finish() != record.metadata.payload_hash.value)) {
    return MatchResult{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  return MatchResult{.code = ::rund::replay::Code::Ok};
}

} // namespace rund::node::replay_detail::payload
