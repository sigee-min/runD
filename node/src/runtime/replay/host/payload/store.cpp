#include "store.hpp"
#include "backend.hpp"
#include "hash.hpp"
#include <kernel/core/checked.hpp>

#include <rund/counter.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rund::node::replay_detail::payload {
namespace {

[[nodiscard]] bool FitsBudget(const std::uint64_t current,
                              const std::uint64_t added,
                              const std::uint64_t budget) noexcept {
  std::uint64_t total = 0u;
  return rund::kernel::checked::add(current, added, total) && total <= budget;
}

[[nodiscard]] bool FitsU32(const std::size_t value) noexcept {
  return value <= std::numeric_limits<std::uint32_t>::max();
}

} // namespace

std::optional<Limits> Limits::runtime(const std::uint32_t hosts,
                                      const std::uint32_t inputs,
                                      const std::uint64_t bytes) noexcept {
  const std::uint64_t records = static_cast<std::uint64_t>(hosts) + inputs;
  const std::uint64_t pieces = records + bytes / kChunkBytes;
  const std::uint64_t staged =
      bytes == 0u ? 0u : 1u + (bytes - 1u) / kChunkBytes;
  if (records > std::numeric_limits<std::uint32_t>::max() ||
      pieces > std::numeric_limits<std::uint32_t>::max() ||
      staged > std::numeric_limits<std::uint32_t>::max() ||
      bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  return Limits{.hosts = hosts,
                .inputs = inputs,
                .pieces = static_cast<std::size_t>(pieces),
                .blobs = static_cast<std::size_t>(pieces),
                .staged = static_cast<std::size_t>(staged),
                .bytes = bytes};
}

std::optional<Limits> Limits::archive(
    const ::rund::node::replay_detail::payload::Archive &archive) noexcept {
  Limits limits{};
  for (const ::rund::node::replay_detail::payload::ArchiveRecord &record :
       archive.records) {
    switch (record.metadata.role) {
    case ::rund::node::replay_detail::payload::Role::Host:
      ++limits.hosts;
      break;
    case ::rund::node::replay_detail::payload::Role::Input:
      ++limits.inputs;
      break;
    default:
      return std::nullopt;
    }
    if (record.pieces.size() >
        std::numeric_limits<std::size_t>::max() - limits.pieces) {
      return std::nullopt;
    }
    limits.pieces += record.pieces.size();
    if (!rund::kernel::checked::add(
            limits.bytes, record.metadata.completed_bytes, limits.bytes)) {
      return std::nullopt;
    }
  }
  limits.blobs = archive.chunks.size();
  if (!FitsU32(limits.hosts + limits.inputs) || !FitsU32(limits.pieces) ||
      !FitsU32(limits.blobs)) {
    return std::nullopt;
  }
  return limits;
}

Store::Store() : backend_{std::make_unique<Backend>()} {}

Store::Store(::rund::replay::Storage storage, const Limits limits,
             ::rund::replay::Diagnostic diagnostic)
    : storage_{std::move(storage)}, limits_{limits},
      staged_index_{limits.staged},
      backend_{std::make_unique<Backend>(storage_, limits.blobs)},
      diagnostic_{diagnostic} {
  if (limits_.bytes > storage_.max_bytes ||
      !FitsU32(limits_.hosts + limits_.inputs) || !FitsU32(limits_.pieces) ||
      !FitsU32(limits_.blobs) || !FitsU32(limits_.staged) ||
      limits_.staged > limits_.pieces || limits_.blobs > limits_.pieces) {
    throw std::invalid_argument{"replay_payload_store_limits_invalid"};
  }
  records_.reserve(limits_.hosts + limits_.inputs);
  host_record_indices_.reserve(limits_.hosts);
  input_record_indices_.reserve(limits_.inputs);
  pieces_.reserve(limits_.pieces);
  piece_scratch_.reserve(limits_.staged);
  staged_blobs_.reserve(limits_.staged);
}

Store::~Store() = default;

Store::Store(Store &&) noexcept = default;

Store &Store::operator=(Store &&) noexcept = default;

bool Store::Append(const std::uint64_t event_sequence,
                   const ::rund::host::EventKind kind, const Capture payload) {
  if (!payload) {
    return false;
  }
  const std::span<const std::byte> bytes = payload.bytes();
  return AppendRecord(
      StoredRecord{
          .metadata =
              {
                  .role = ::rund::node::replay_detail::payload::Role::Host,
                  .event_sequence = event_sequence,
                  .kind = kind,
                  .completed_bytes = bytes.size(),
                  .payload_hash = payload.hash(),
              },
      },
      payload);
}

bool Store::Append(const std::uint64_t event_sequence,
                   const ::rund::host::EventKind kind,
                   ::rund::node::replay_detail::payload::Bytes bytes,
                   const Capture payload) {
  if (!payload) {
    return false;
  }
  const std::span<const std::byte> span = bytes.span();
  return AppendRecord(
      StoredRecord{
          .metadata =
              {
                  .role = ::rund::node::replay_detail::payload::Role::Host,
                  .event_sequence = event_sequence,
                  .kind = kind,
                  .completed_bytes = span.size(),
                  .payload_hash = payload.hash(),
              },
      },
      payload, std::move(bytes));
}

bool Store::AppendInput(const std::uint64_t source, const std::uint64_t schema,
                        const std::uint64_t sequence,
                        const InputSourceRange source_range,
                        ::rund::node::replay_detail::payload::Bytes bytes,
                        const Capture payload) {
  if (source == 0u || schema == 0u || !payload) {
    return false;
  }
  const std::span<const std::byte> span = bytes.span();
  return AppendRecord(
      StoredRecord{
          .metadata =
              {
                  .role = ::rund::node::replay_detail::payload::Role::Input,
                  .input_source = source,
                  .input_schema = schema,
                  .input_sequence = sequence,
                  .source_event_offset = source_range.event_offset,
                  .source_event_count = source_range.event_count,
                  .source_payload_offset = source_range.payload_offset,
                  .source_payload_count = source_range.payload_count,
                  .source_hash = source_range.hash,
                  .completed_bytes = span.size(),
                  .payload_hash = payload.hash(),
              },
      },
      payload, std::move(bytes));
}

bool Store::CapturesIngress() const noexcept { return diagnostic_.enabled(); }

::rund::StableHash Store::CaptureIngress(const std::uint64_t event_sequence,
                                         const ::rund::host::EventKind kind,
                                         const RawByteSource &source) noexcept {
  loaded_diagnostic_active_ = false;
  loaded_diagnostic_ = {};
  return diagnostic_.Capture(event_sequence, kind, source);
}

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
  if (!FitsBudget(logical_bytes_, logical_added, limits_.bytes)) {
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
  if (!FitsBudget(backend_->encoded_bytes(), staged_encoded_bytes,
                  limits_.bytes)) {
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

void Store::Commit(StoredRecord record, const std::uint64_t logical_bytes) {
  records_.push_back(std::move(record));
  const std::size_t index = records_.size() - 1u;
  if (records_.back().metadata.role ==
      ::rund::node::replay_detail::payload::Role::Input) {
    input_record_indices_.push_back(index);
  } else {
    host_record_indices_.push_back(index);
  }
  logical_bytes_ += logical_bytes;
}

std::uint64_t Store::logical_bytes() const noexcept { return logical_bytes_; }

std::uint64_t Store::retained_bytes() const noexcept {
  return backend_->retained_bytes();
}

std::uint64_t Store::encoded_bytes() const noexcept {
  return backend_->encoded_bytes();
}

std::uint64_t Store::segment_count() const noexcept {
  return backend_->segment_count();
}

const ::rund::replay::Storage &Store::storage() const noexcept {
  return storage_;
}

const Limits &Store::limits() const noexcept { return limits_; }

const std::vector<StoredRecord> &Store::records() const noexcept {
  return records_;
}

std::span<const Piece>
Store::pieces(const StoredRecord &record) const noexcept {
  const std::size_t offset = record.piece_offset;
  const std::size_t count = record.piece_count;
  if (offset > pieces_.size() || count > pieces_.size() - offset) {
    return {};
  }
  return std::span<const Piece>{pieces_}.subspan(offset, count);
}

std::size_t Store::host_record_count() const noexcept {
  return host_record_indices_.size();
}

std::size_t Store::input_record_count() const noexcept {
  return input_record_indices_.size();
}

std::size_t Store::host_record_index(const std::size_t index) const noexcept {
  return index < host_record_indices_.size()
             ? host_record_indices_[index]
             : std::numeric_limits<std::size_t>::max();
}

std::size_t Store::input_record_index(const std::size_t index) const noexcept {
  return index < input_record_indices_.size()
             ? input_record_indices_[index]
             : std::numeric_limits<std::size_t>::max();
}

const std::vector<Blob> &Store::blobs() const noexcept {
  return backend_->blobs();
}

void Store::Clear() noexcept {
  records_.clear();
  host_record_indices_.clear();
  input_record_indices_.clear();
  pieces_.clear();
  piece_scratch_.clear();
  staged_blobs_.clear();
  staged_index_.clear();
  backend_->Clear();
  diagnostic_.Clear();
  loaded_diagnostic_ = {};
  publication_ = {};
  publication_chunks_ = 0u;
  logical_bytes_ = 0u;
  growths_ = 0u;
  loaded_payload_hash_ = 0u;
  loaded_archive_ = false;
  loaded_diagnostic_active_ = false;
}

void Store::RebuildRoleIndices() {
  host_record_indices_.clear();
  input_record_indices_.clear();
  for (std::size_t index = 0u; index < records_.size(); ++index) {
    if (records_[index].metadata.role ==
        ::rund::node::replay_detail::payload::Role::Input) {
      input_record_indices_.push_back(index);
    } else {
      host_record_indices_.push_back(index);
    }
  }
}

} // namespace rund::node::replay_detail::payload
