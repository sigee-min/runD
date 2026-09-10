#include "../store.hpp"

#include "../backend.hpp"

#include <limits>
#include <utility>

namespace rund::node::replay_detail::payload {

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
