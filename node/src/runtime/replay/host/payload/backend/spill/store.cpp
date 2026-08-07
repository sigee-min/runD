#include "local.hpp"

#include "../../../../exception.hpp"
#include "../../codec.hpp"
#include "../../hash.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace rund::node::replay_detail::payload {

Spill::Spill() = default;

Spill::Spill(const ::rund::replay::Storage &storage, const std::size_t capacity)
    : storage_{storage}, cache_{storage.cached_bytes} {
  blobs_.reserve(capacity);
  refs_.reserve(capacity);
}

AppendResult Spill::Append(Blob blob) {
  if (!rund::kernel::checked::add(spill::kHeaderBytes, blob.encoded_bytes)) {
    return AppendResult{.code =
                            ::rund::replay::Code::HostPayloadCapacityExceeded};
  }
  const std::uint32_t blob_index = static_cast<std::uint32_t>(blobs_.size());
  const std::uint64_t record_bytes = spill::kHeaderBytes + blob.encoded_bytes;
  if (record_bytes > storage_.segment_bytes) {
    return AppendResult{.code =
                            ::rund::replay::Code::HostPayloadSegmentTooSmall};
  }
  if (current_segment_bytes_ != 0u &&
      (!rund::kernel::checked::add(current_segment_bytes_, record_bytes) ||
       current_segment_bytes_ + record_bytes > storage_.segment_bytes)) {
    if (current_segment_index_ == std::numeric_limits<std::uint32_t>::max()) {
      return AppendResult{
          .code = ::rund::replay::Code::HostPayloadCapacityExceeded};
    }
    ++current_segment_index_;
    current_segment_bytes_ = 0u;
  }
  if (!EnsureGeneration()) {
    return AppendResult{.code =
                            ::rund::replay::Code::HostPayloadSpillWriteFailed};
  }

  const std::optional<spill::Space> space =
      spill::inspect(generation_->directory(), current_segment_bytes_,
                     record_bytes, storage_.minimum_free_bytes);
  if (!space.has_value()) {
    return AppendResult{.code =
                            ::rund::replay::Code::HostPayloadSpillWriteFailed};
  }
  ::rund::storage::Reservation reservation =
      generation_->Reserve(space->allocation);
  if (!reservation) {
    return AppendResult{.code =
                            ::rund::replay::Code::HostPayloadCapacityExceeded};
  }
  if (!space->headroom) {
    return AppendResult{.code =
                            ::rund::replay::Code::HostPayloadSpillWriteFailed};
  }
  if (!generation_->Stage(std::move(reservation))) {
    return AppendResult{.code =
                            ::rund::replay::Code::HostPayloadSpillWriteFailed};
  }
  if (!spill::write(generation_, current_segment_index_, current_segment_bytes_,
                    blob_index, blob) ||
      !generation_->CommitLast(
          ::rund::storage::Usage{.physical_bytes = record_bytes,
                                 .allocated_bytes = space->allocation})) {
    return AppendResult{.code =
                            ::rund::replay::Code::HostPayloadSpillWriteFailed};
  }

  const std::size_t refs_capacity = refs_.capacity();
  const std::size_t blobs_capacity = blobs_.capacity();
  refs_.push_back(SpillRef{
      .segment_index = current_segment_index_,
      .segment_offset = current_segment_bytes_,
      .record_bytes = record_bytes,
  });
  if (refs_.capacity() != refs_capacity) {
    growths_ = ::rund::detail::counter::SaturatingAdd(growths_, 1u);
  }
  current_segment_bytes_ += record_bytes;
  encoded_bytes_ += blob.encoded_bytes;
  Blob stored = std::move(blob);
  stored.encoded = {};
  blobs_.push_back(std::move(stored));
  if (blobs_.capacity() != blobs_capacity) {
    growths_ = ::rund::detail::counter::SaturatingAdd(growths_, 1u);
  }
  return AppendResult{.code = ::rund::replay::Code::Ok,
                      .blob_index = blob_index,
                      .encoded_bytes = record_bytes};
}

bool Spill::CanAppend(const Blob &blob) const noexcept {
  return rund::kernel::checked::add(spill::kHeaderBytes, blob.encoded_bytes) &&
         spill::kHeaderBytes + blob.encoded_bytes <= storage_.segment_bytes;
}

SpillMark Spill::Mark() const noexcept {
  return SpillMark{.blobs = blobs_.size(),
                   .refs = refs_.size(),
                   .reservations = generation_ == nullptr
                                       ? 0u
                                       : generation_->reservation_count(),
                   .encoded_bytes = encoded_bytes_,
                   .segment = current_segment_index_,
                   .segment_bytes = current_segment_bytes_};
}

bool Spill::Rollback(const SpillMark mark) noexcept {
  const std::uint32_t last_segment = current_segment_index_;
  bool clean = true;
  std::error_code error{};
  if (last_segment > mark.segment || mark.segment_bytes == 0u) {
    const std::uint32_t first_removed =
        mark.segment_bytes == 0u ? mark.segment : mark.segment + 1u;
    for (std::uint32_t segment = last_segment;; --segment) {
      std::filesystem::remove(spill::path(generation_, segment), error);
      clean = clean && !error;
      error.clear();
      if (segment == first_removed) {
        break;
      }
    }
  }
  if (mark.segment_bytes != 0u) {
    std::filesystem::resize_file(spill::path(generation_, mark.segment),
                                 mark.segment_bytes, error);
    clean = clean && !error;
  }
  blobs_.resize(mark.blobs);
  refs_.resize(mark.refs);
  encoded_bytes_ = mark.encoded_bytes;
  current_segment_index_ = mark.segment;
  current_segment_bytes_ = mark.segment_bytes;
  if (clean && generation_ != nullptr) {
    generation_->Rollback(mark.reservations);
  }
  return clean;
}

ReadResult Spill::Read(const std::uint32_t blob_index) const {
  if (blob_index >= refs_.size() || blob_index >= blobs_.size()) {
    return ReadResult{.code = ::rund::replay::Code::HostPayloadMissing,
                      .bytes = {}};
  }
  std::vector<std::byte> cached{};
  if (cache_.Get(blob_index, cached)) {
    return ReadResult{.code = ::rund::replay::Code::Ok,
                      .bytes = std::move(cached)};
  }
  spill::Segment segment = spill::read(generation_, refs_, blobs_, blob_index);
  if (!segment.ok()) {
    return ReadResult{.code = segment.code, .bytes = {}};
  }
  Blob &blob = segment.blob;
  ReadResult decoded_result = Decode(blob.payload_hash, blob.uncompressed_bytes,
                                     blob.codec, blob.encoded.span());
  if (decoded_result.ok()) {
    cache_.Put(blob_index, decoded_result.bytes);
  }
  return decoded_result;
}

EncodedResult Spill::Encoded(const std::uint32_t blob_index) const noexcept {
  try {
    spill::Segment segment =
        spill::read(generation_, refs_, blobs_, blob_index);
    if (!segment.ok()) {
      return EncodedResult{.code = segment.code};
    }
    return EncodedResult{.code = ::rund::replay::Code::Ok,
                         .bytes = std::move(segment.blob.encoded)};
  } catch (...) {
    return EncodedResult{
        .code = ::rund::node::replay_detail::CurrentExceptionCode({
            .bad_alloc = ::rund::replay::Code::AllocationFailed,
            .length_error = ::rund::replay::Code::HostPayloadMissing,
            .unexpected = ::rund::replay::Code::HostPayloadMissing,
        })};
  }
}

ReadStatus Spill::ReadInto(const std::uint32_t blob_index,
                           const std::span<std::byte> output,
                           ByteHash *const record_hash) const {
  if (blob_index >= refs_.size() || blob_index >= blobs_.size()) {
    return ReadStatus{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  const Blob &expected = blobs_[blob_index];
  if (expected.uncompressed_bytes != output.size()) {
    return ReadStatus{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  if (const std::optional<bool> hit =
          cache_.Read(blob_index, output, record_hash);
      hit.has_value()) {
    return *hit ? ReadStatus{.code = ::rund::replay::Code::Ok}
                : ReadStatus{.code =
                                 ::rund::replay::Code::HostPayloadHashInvalid};
  }
  spill::Segment segment = spill::read(generation_, refs_, blobs_, blob_index);
  if (!segment.ok()) {
    return ReadStatus{.code = segment.code};
  }
  const Blob &blob = segment.blob;
  std::vector<std::byte> decoded_bytes(output.size());
  const ReadStatus decoded =
      DecodeInto(blob.payload_hash, blob.uncompressed_bytes, blob.codec,
                 blob.encoded.span(), decoded_bytes, record_hash);
  if (!decoded.ok()) {
    return decoded;
  }
  std::copy(decoded_bytes.begin(), decoded_bytes.end(), output.begin());
  cache_.Put(blob_index, std::move(decoded_bytes));
  return ReadStatus{.code = ::rund::replay::Code::Ok};
}

ReadStatus Spill::Matches(const std::uint32_t blob_index,
                          const std::span<const std::byte> expected,
                          ByteHash *const record_hash) const {
  if (blob_index >= refs_.size() || blob_index >= blobs_.size()) {
    return ReadStatus{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  if (const std::optional<bool> cached =
          cache_.Matches(blob_index, expected, record_hash);
      cached.has_value()) {
    return *cached ? ReadStatus{.code = ::rund::replay::Code::Ok}
                   : ReadStatus{
                         .code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  spill::Segment segment = spill::read(generation_, refs_, blobs_, blob_index);
  if (!segment.ok()) {
    return ReadStatus{.code = segment.code};
  }
  const Blob &blob = segment.blob;
  return VerifiedMatches(blob, expected, blob.payload_hash, record_hash)
             ? ReadStatus{.code = ::rund::replay::Code::Ok}
             : ReadStatus{.code = ::rund::replay::Code::HostPayloadHashInvalid};
}

std::uint64_t Spill::retained_bytes() const noexcept {
  return cache_.Report().cached_bytes;
}

std::uint64_t Spill::encoded_bytes() const noexcept { return encoded_bytes_; }

const std::vector<Blob> &Spill::blobs() const noexcept { return blobs_; }

std::uint64_t Spill::segment_count() const noexcept {
  if (blobs_.empty()) {
    return 0u;
  }
  return static_cast<std::uint64_t>(current_segment_index_) + 1u;
}

const std::vector<SpillRef> &Spill::refs() const noexcept { return refs_; }

::rund::replay::StorageReport Spill::Report() const noexcept {
  ::rund::replay::StorageReport report = cache_.Report();
  report.mode = ::rund::replay::StorageMode::Spill;
  report.encoded_bytes = encoded_bytes_;
  report.growths =
      ::rund::detail::counter::SaturatingAdd(report.growths, growths_);
  report.chunk_count = static_cast<std::uint64_t>(blobs_.size());
  report.segment_count = segment_count();
  report.physical_bytes =
      encoded_bytes_ +
      spill::kHeaderBytes * static_cast<std::uint64_t>(blobs_.size());
  if (generation_ != nullptr) {
    const ::rund::storage::Usage usage = generation_->usage();
    report.allocated_bytes = usage.allocated_bytes;
    report.reserved_bytes = generation_->reserved_bytes();
  }
  return report;
}

const std::shared_ptr<const SpillGeneration> &
Spill::generation() const noexcept {
  return generation_;
}

void Spill::LoadArchive(
    std::vector<Blob> blobs, std::vector<SpillRef> refs,
    const std::uint64_t encoded_bytes,
    std::shared_ptr<const SpillGeneration> generation) noexcept {
  blobs_ = std::move(blobs);
  refs_ = std::move(refs);
  generation_ = std::move(generation);
  encoded_bytes_ = encoded_bytes;
  current_segment_index_ = 0u;
  current_segment_bytes_ = 0u;
  cache_.Clear();
  growths_ = 0u;
  for (const SpillRef &ref : refs_) {
    if (ref.segment_index > current_segment_index_) {
      current_segment_index_ = ref.segment_index;
      current_segment_bytes_ = 0u;
    }
    if (ref.segment_index == current_segment_index_) {
      current_segment_bytes_ = std::max(current_segment_bytes_,
                                        ref.segment_offset + ref.record_bytes);
    }
  }
}

void Spill::Clear() noexcept {
  blobs_.clear();
  refs_.clear();
  generation_.reset();
  encoded_bytes_ = 0u;
  current_segment_index_ = 0u;
  current_segment_bytes_ = 0u;
  cache_.Clear();
  growths_ = 0u;
}

bool Spill::EnsureGeneration() noexcept {
  if (generation_ != nullptr) {
    return true;
  }
  generation_ = SpillGeneration::Create(storage_.directory, storage_.budget,
                                        blobs_.capacity());
  return generation_ != nullptr;
}

} // namespace rund::node::replay_detail::payload
