#include "../backend.hpp"

#include <rund/counter.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace rund::node::replay_detail::payload {

Memory::Memory(const std::size_t capacity) { blobs_.reserve(capacity); }

std::uint32_t Memory::Append(Blob blob) {
  const std::uint32_t index = static_cast<std::uint32_t>(blobs_.size());
  const std::size_t capacity = blobs_.capacity();
  retained_bytes_ += blob.encoded_bytes;
  encoded_bytes_ += blob.encoded_bytes;
  blobs_.push_back(std::move(blob));
  if (blobs_.capacity() != capacity) {
    growths_ = ::rund::detail::counter::SaturatingAdd(growths_, 1u);
  }
  return index;
}

bool Memory::CanAppend(const Blob & /*blob*/) const noexcept { return true; }

ReadResult Memory::Read(const std::uint32_t blob_index) const {
  if (blob_index >= blobs_.size()) {
    return ReadResult{.code = ::rund::replay::Code::HostPayloadMissing,
                      .bytes = {}};
  }
  const Blob &blob = blobs_[blob_index];
  return Decode(blob.payload_hash, blob.uncompressed_bytes, blob.codec,
                blob.encoded.span());
}

EncodedResult Memory::Encoded(const std::uint32_t blob_index) const noexcept {
  if (blob_index >= blobs_.size()) {
    return EncodedResult{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  const Blob &blob = blobs_[blob_index];
  if (blob.encoded.size() != blob.encoded_bytes) {
    return EncodedResult{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  return EncodedResult{.code = ::rund::replay::Code::Ok, .bytes = blob.encoded};
}

ReadStatus Memory::ReadInto(const std::uint32_t blob_index,
                            const std::span<std::byte> output,
                            ByteHash *const record_hash) const noexcept {
  if (blob_index >= blobs_.size()) {
    return ReadStatus{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  const Blob &blob = blobs_[blob_index];
  return DecodeInto(blob.payload_hash, blob.uncompressed_bytes, blob.codec,
                    blob.encoded.span(), output, record_hash);
}

std::uint64_t Memory::retained_bytes() const noexcept {
  return retained_bytes_;
}

std::uint64_t Memory::encoded_bytes() const noexcept { return encoded_bytes_; }

std::uint64_t Memory::growths() const noexcept { return growths_; }

const std::vector<Blob> &Memory::blobs() const noexcept { return blobs_; }

MemoryMark Memory::Mark() const noexcept {
  return MemoryMark{.blobs = blobs_.size(),
                    .retained_bytes = retained_bytes_,
                    .encoded_bytes = encoded_bytes_};
}

void Memory::Rollback(const MemoryMark mark) noexcept {
  blobs_.resize(mark.blobs);
  retained_bytes_ = mark.retained_bytes;
  encoded_bytes_ = mark.encoded_bytes;
}

void Memory::Clear() noexcept {
  blobs_.clear();
  retained_bytes_ = 0u;
  encoded_bytes_ = 0u;
  growths_ = 0u;
}

} // namespace rund::node::replay_detail::payload
