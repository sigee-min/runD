#include "backend.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace rund::node::replay_detail::payload {

Backend::Backend() : Backend(::rund::replay::Storage{}, 0u) {}

Backend::Backend(const ::rund::replay::Storage &storage,
                 const std::size_t capacity)
    : storage_{storage}, memory_{capacity}, spill_{storage, capacity},
      index_{capacity} {}

std::optional<std::uint32_t>
Backend::Find(const std::span<const std::byte> bytes,
              const ::rund::StableHash payload_hash) const {
  return index_.find(payload_hash.value, [&](const std::uint32_t blob_index) {
    if (blob_index >= blobs().size()) {
      return false;
    }
    switch (storage_.mode) {
    case ::rund::replay::StorageMode::Memory:
      return ::rund::node::replay_detail::payload::Matches(blobs()[blob_index],
                                                           bytes, payload_hash);
    case ::rund::replay::StorageMode::Spill:
      return spill_.Matches(blob_index, bytes).ok();
    }
    return false;
  });
}

BatchResult Backend::Append(std::vector<Blob> &blobs) {
  if (!CanAppend(blobs)) {
    return BatchResult{
        .code = writable_ ? ::rund::replay::Code::HostPayloadCapacityExceeded
                          : ::rund::replay::Code::HostPayloadBackendPoisoned};
  }
  const std::uint32_t first = static_cast<std::uint32_t>(this->blobs().size());
  const MemoryMark memory_mark = memory_.Mark();
  const SpillMark spill_mark = spill_.Mark();
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory: {
    for (Blob &blob : blobs) {
      static_cast<void>(memory_.Append(std::move(blob)));
    }
    break;
  }
  case ::rund::replay::StorageMode::Spill: {
    for (Blob &blob : blobs) {
      const AppendResult appended = spill_.Append(std::move(blob));
      if (!appended.ok()) {
        if (!spill_.Rollback(spill_mark)) {
          writable_ = false;
        }
        return BatchResult{.code = appended.code};
      }
    }
    break;
  }
  }
  for (std::size_t offset = 0u; offset < blobs.size(); ++offset) {
    const std::uint32_t blob_index = static_cast<std::uint32_t>(first + offset);
    if (!index(blob_index, this->blobs()[blob_index].payload_hash)) {
      switch (storage_.mode) {
      case ::rund::replay::StorageMode::Memory:
        memory_.Rollback(memory_mark);
        break;
      case ::rund::replay::StorageMode::Spill:
        if (!spill_.Rollback(spill_mark)) {
          writable_ = false;
        }
        break;
      }
      if (!rebuild()) {
        writable_ = false;
      }
      return BatchResult{.code =
                             ::rund::replay::Code::HostPayloadIndexExhausted};
    }
  }
  return BatchResult{.code = ::rund::replay::Code::Ok,
                     .first_blob = first,
                     .blob_count = static_cast<std::uint32_t>(blobs.size())};
}

bool Backend::CanAppend(const std::span<const Blob> blobs) const noexcept {
  if (!writable_ || blobs.size() > index_.capacity() - index_.size()) {
    return false;
  }
  for (const Blob &blob : blobs) {
    switch (storage_.mode) {
    case ::rund::replay::StorageMode::Memory:
      if (!memory_.CanAppend(blob)) {
        return false;
      }
      break;
    case ::rund::replay::StorageMode::Spill:
      if (!spill_.CanAppend(blob)) {
        return false;
      }
      break;
    }
  }
  return true;
}

ReadResult Backend::Read(const std::uint32_t blob_index) const {
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    return memory_.Read(blob_index);
  case ::rund::replay::StorageMode::Spill:
    return spill_.Read(blob_index);
  }
  return ReadResult{.code = ::rund::replay::Code::HostStorageBackendUnavailable,
                    .bytes = {}};
}

EncodedResult Backend::Encoded(const std::uint32_t blob_index) const noexcept {
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    return memory_.Encoded(blob_index);
  case ::rund::replay::StorageMode::Spill:
    return spill_.Encoded(blob_index);
  }
  return EncodedResult{.code =
                           ::rund::replay::Code::HostStorageBackendUnavailable};
}

ReadStatus Backend::ReadInto(const std::uint32_t blob_index,
                             const std::span<std::byte> output,
                             ByteHash *const record_hash) const {
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    return memory_.ReadInto(blob_index, output, record_hash);
  case ::rund::replay::StorageMode::Spill:
    return spill_.ReadInto(blob_index, output, record_hash);
  }
  return ReadStatus{.code =
                        ::rund::replay::Code::HostStorageBackendUnavailable};
}

ReadStatus Backend::Matches(const std::uint32_t blob_index,
                            const std::span<const std::byte> expected,
                            ByteHash *const record_hash) const {
  if (blob_index >= blobs().size()) {
    return ReadStatus{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory: {
    const Blob &blob = memory_.blobs()[blob_index];
    return ::rund::node::replay_detail::payload::VerifiedMatches(
               blob, expected, blob.payload_hash, record_hash)
               ? ReadStatus{.code = ::rund::replay::Code::Ok}
               : ReadStatus{.code =
                                ::rund::replay::Code::HostPayloadHashInvalid};
  }
  case ::rund::replay::StorageMode::Spill: {
    return spill_.Matches(blob_index, expected, record_hash);
  }
  }
  return ReadStatus{.code =
                        ::rund::replay::Code::HostStorageBackendUnavailable};
}

std::uint64_t Backend::retained_bytes() const noexcept {
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    return memory_.retained_bytes();
  case ::rund::replay::StorageMode::Spill:
    return spill_.retained_bytes();
  }
  return 0u;
}

std::uint64_t Backend::encoded_bytes() const noexcept {
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    return memory_.encoded_bytes();
  case ::rund::replay::StorageMode::Spill:
    return spill_.encoded_bytes();
  }
  return 0u;
}

const std::vector<Blob> &Backend::blobs() const noexcept {
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    return memory_.blobs();
  case ::rund::replay::StorageMode::Spill:
    return spill_.blobs();
  }
  return memory_.blobs();
}

std::uint64_t Backend::segment_count() const noexcept {
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    return 0u;
  case ::rund::replay::StorageMode::Spill:
    return spill_.segment_count();
  }
  return 0u;
}

const std::vector<SpillRef> &Backend::refs() const noexcept {
  return spill_.refs();
}

::rund::replay::StorageReport Backend::Report() const noexcept {
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    return ::rund::replay::StorageReport{
        .mode = ::rund::replay::StorageMode::Memory,
        .encoded_bytes = memory_.encoded_bytes(),
        .cached_bytes = memory_.retained_bytes(),
        .growths = memory_.growths(),
        .chunk_count = static_cast<std::uint64_t>(memory_.blobs().size()),
        .segment_count = 0u,
    };
  case ::rund::replay::StorageMode::Spill:
    return spill_.Report();
  }
  return ::rund::replay::StorageReport{};
}

const std::shared_ptr<const SpillGeneration> &
Backend::generation() const noexcept {
  return spill_.generation();
}

void Backend::LoadArchive(
    std::vector<Blob> blobs, std::vector<SpillRef> refs,
    const std::uint64_t encoded_bytes,
    std::shared_ptr<const SpillGeneration> generation) noexcept {
  Clear();
  switch (storage_.mode) {
  case ::rund::replay::StorageMode::Memory:
    for (Blob &blob : blobs) {
      static_cast<void>(memory_.Append(std::move(blob)));
    }
    break;
  case ::rund::replay::StorageMode::Spill:
    spill_.LoadArchive(std::move(blobs), std::move(refs), encoded_bytes,
                       std::move(generation));
    break;
  }
  for (std::size_t index = 0u; index < this->blobs().size(); ++index) {
    if (!this->index(static_cast<std::uint32_t>(index),
                     this->blobs()[index].payload_hash)) {
      Clear();
      return;
    }
  }
}

void Backend::Clear() noexcept {
  memory_.Clear();
  spill_.Clear();
  index_.clear();
}

bool Backend::index(const std::uint32_t blob_index,
                    const ::rund::StableHash payload_hash) noexcept {
  return index_.insert(payload_hash.value, blob_index);
}

bool Backend::rebuild() noexcept {
  index_.clear();
  for (std::size_t blob = 0u; blob < blobs().size(); ++blob) {
    if (!index(static_cast<std::uint32_t>(blob), blobs()[blob].payload_hash)) {
      return false;
    }
  }
  return true;
}

} // namespace rund::node::replay_detail::payload
