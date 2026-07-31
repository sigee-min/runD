#include "ring.hpp"
#include <kernel/core/checked.hpp>

#include "../../../../../host/hash/bytes.hpp"
#include "../../../../../host/hash/fields.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace rund::node::replay_detail::payload {
namespace {

[[nodiscard]] bool IngressKind(const ::rund::host::EventKind kind) noexcept {
  return kind == ::rund::host::EventKind::NetRecv ||
         kind == ::rund::host::EventKind::NetRecvDatagram ||
         kind == ::rund::host::EventKind::NetRecvVectored;
}

[[nodiscard]] constexpr std::uint64_t
Wrap(const std::uint64_t base, const std::uint64_t offset,
     const std::size_t capacity) noexcept {
  const std::uint64_t bound = static_cast<std::uint64_t>(capacity);
  const std::uint64_t remaining = bound - base;
  return offset >= remaining ? offset - remaining : base + offset;
}

static_assert(Wrap(0u, 0u, 8u) == 0u);
static_assert(Wrap(3u, 4u, 8u) == 7u);
static_assert(Wrap(7u, 1u, 8u) == 0u);
static_assert(Wrap(7u, 8u, 8u) == 7u);

[[nodiscard]] std::uint64_t
DiagnosticHash(const ::rund::node::replay_detail::payload::DiagnosticArchive
                   &archive) noexcept {
  if (archive.records.empty() && archive.bytes.empty() &&
      archive.report.retained_bytes == 0u &&
      archive.report.retained_records == 0u &&
      archive.report.evicted_records == 0u &&
      archive.report.dropped_records == 0u) {
    return 0u;
  }
  host_detail::StableHashState hash{};
  hash.Mix(archive.report.retained_bytes);
  hash.Mix(archive.report.retained_records);
  hash.Mix(archive.report.evicted_records);
  hash.Mix(archive.report.dropped_records);
  hash.Mix(static_cast<std::uint64_t>(archive.records.size()));
  for (const ::rund::node::replay_detail::payload::DiagnosticRecord &record :
       archive.records) {
    hash.Mix(static_cast<std::uint8_t>(record.role));
    hash.Mix(record.event_sequence);
    hash.Mix(static_cast<std::uint16_t>(record.kind));
    hash.Mix(record.offset);
    hash.Mix(record.byte_count);
    hash.Mix(record.payload_hash.value);
    if (record.offset <= archive.bytes.size() &&
        record.byte_count <= archive.bytes.size() - record.offset) {
      for (const std::byte value : archive.bytes.span().subspan(
               static_cast<std::size_t>(record.offset),
               static_cast<std::size_t>(record.byte_count))) {
        hash.Mix(std::to_integer<std::uint8_t>(value));
      }
    }
  }
  return hash.Finish().value;
}

} // namespace

RawCaptureRing::RawCaptureRing(const ::rund::replay::Diagnostic config)
    : config_(config) {
  if (config_.window_bytes == 0u || config_.window_records == 0u ||
      config_.window_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      config_.window_records >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    config_ = {};
    return;
  }
  bytes_.resize(static_cast<std::size_t>(config_.window_bytes));
  records_.resize(static_cast<std::size_t>(config_.window_records));
}

void RawCaptureRing::EvictOldest() noexcept {
  if (record_count_ == 0u) {
    return;
  }
  const Record &record = records_[static_cast<std::size_t>(record_head_)];
  if (!bytes_.empty()) {
    byte_head_ = Wrap(record.byte_offset, record.byte_count, bytes_.size());
  }
  byte_size_ -= record.byte_count;
  record_head_ = Wrap(record_head_, 1u, records_.size());
  --record_count_;
  evicted_records_ =
      ::rund::detail::counter::SaturatingAdd(evicted_records_, 1u);
}

::rund::StableHash HashIngress(const RawByteSource &source) noexcept {
  if (source.byte_count == 0u) {
    return host_detail::StableByteHash(nullptr, 0u);
  }
  if (source.slice == nullptr || source.context == nullptr ||
      source.slice_count == 0u || source.byte_count > source.admitted_bytes) {
    return host_detail::StableByteHash(
        nullptr, static_cast<std::size_t>(source.byte_count));
  }

  std::uint64_t remaining = source.byte_count;
  const std::byte *first = nullptr;
  std::size_t first_size = 0u;
  bool have_first = false;
  for (std::size_t index = 0u; index < source.slice_count; ++index) {
    const std::span<const std::byte> slice =
        source.slice(source.context, index);
    const std::size_t size = static_cast<std::size_t>(
        std::min<std::uint64_t>(slice.size(), remaining));
    if (size != 0u && slice.data() == nullptr) {
      return host_detail::StableByteHash(
          nullptr, static_cast<std::size_t>(source.byte_count));
    }
    remaining -= size;
    if (size == 0u) {
      if (remaining == 0u) {
        break;
      }
      continue;
    }
    if (!have_first) {
      first = slice.data();
      first_size = size;
      have_first = true;
      if (remaining == 0u) {
        break;
      }
      continue;
    }

    host_detail::StableByteHasher hash{};
    hash.Append({first, first_size});
    hash.Append(slice.first(size));
    for (++index; remaining != 0u && index < source.slice_count; ++index) {
      const std::span<const std::byte> tail =
          source.slice(source.context, index);
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(tail.size(), remaining));
      if (count != 0u && tail.data() == nullptr) {
        return host_detail::StableByteHash(
            nullptr, static_cast<std::size_t>(source.byte_count));
      }
      hash.Append(tail.first(count));
      remaining -= count;
    }
    return remaining == 0u
               ? hash.Finish()
               : host_detail::StableByteHash(
                     nullptr, static_cast<std::size_t>(source.byte_count));
  }
  return remaining == 0u
             ? host_detail::StableByteHash(first, first_size)
             : host_detail::StableByteHash(
                   nullptr, static_cast<std::size_t>(source.byte_count));
}

::rund::StableHash
RawCaptureRing::Retain(const std::uint64_t event_sequence,
                       const ::rund::host::EventKind kind,
                       const RawByteSource &source) noexcept {
  while (record_count_ == records_.size() ||
         source.byte_count > bytes_.size() - byte_size_) {
    EvictOldest();
  }
  const std::uint64_t offset = Wrap(byte_head_, byte_size_, bytes_.size());
  std::size_t tail = static_cast<std::size_t>(offset);
  std::uint64_t remaining = source.byte_count;
  host_detail::StableByteHasher hash{};
  for (std::size_t index = 0u; index < source.slice_count && remaining != 0u;
       ++index) {
    const std::span<const std::byte> slice =
        source.slice(source.context, index);
    const std::size_t size = static_cast<std::size_t>(
        std::min<std::uint64_t>(slice.size(), remaining));
    if (size == 0u) {
      continue;
    }
    if (slice.data() == nullptr) {
      return host_detail::StableByteHash(
          nullptr, static_cast<std::size_t>(source.byte_count));
    }
    const std::size_t first = std::min(size, bytes_.size() - tail);
    std::memcpy(bytes_.data() + tail, slice.data(), first);
    hash.Append(std::span<const std::byte>{bytes_.data() + tail, first});
    if (first != size) {
      const std::size_t second = size - first;
      std::memcpy(bytes_.data(), slice.data() + first, second);
      hash.Append(std::span<const std::byte>{bytes_.data(), second});
    }
    tail = static_cast<std::size_t>(Wrap(tail, size, bytes_.size()));
    remaining -= size;
  }
  if (remaining != 0u) {
    return host_detail::StableByteHash(
        nullptr, static_cast<std::size_t>(source.byte_count));
  }
  const ::rund::StableHash payload_hash = hash.Finish();
  const std::uint64_t slot = Wrap(record_head_, record_count_, records_.size());
  records_[static_cast<std::size_t>(slot)] = Record{
      .event_sequence = event_sequence,
      .kind = kind,
      .byte_offset = offset,
      .byte_count = source.byte_count,
      .payload_hash = payload_hash,
  };
  byte_size_ += source.byte_count;
  ++record_count_;
  return payload_hash;
}

bool RawCaptureRing::enabled() const noexcept {
  return !bytes_.empty() && !records_.empty();
}

::rund::StableHash
RawCaptureRing::Capture(const std::uint64_t event_sequence,
                        const ::rund::host::EventKind kind,
                        const RawByteSource &source) noexcept {
  if (!enabled()) {
    return HashIngress(source);
  }
  if (source.byte_count == 0u) {
    return host_detail::StableByteHash(nullptr, 0u);
  }
  if (!IngressKind(kind) || event_sequence == 0u || source.slice == nullptr ||
      source.context == nullptr || source.slice_count == 0u ||
      source.byte_count > source.admitted_bytes ||
      source.byte_count > bytes_.size()) {
    dropped_records_ =
        ::rund::detail::counter::SaturatingAdd(dropped_records_, 1u);
    return HashIngress(source);
  }
  return Retain(event_sequence, kind, source);
}

void RawCaptureRing::Copy(const std::uint64_t offset,
                          const std::span<std::byte> output) const noexcept {
  if (output.empty()) {
    return;
  }
  const std::size_t begin = static_cast<std::size_t>(offset);
  const std::size_t first = std::min(output.size(), bytes_.size() - begin);
  std::memcpy(output.data(), bytes_.data() + begin, first);
  if (first != output.size()) {
    std::memcpy(output.data() + first, bytes_.data(), output.size() - first);
  }
}

::rund::node::replay_detail::payload::DiagnosticArchive
RawCaptureRing::Archive() const {
  ::rund::node::replay_detail::payload::DiagnosticArchive archive{};
  archive.report = ::rund::replay::DiagnosticReport{
      .retained_bytes = byte_size_,
      .retained_records = record_count_,
      .evicted_records = evicted_records_,
      .dropped_records = dropped_records_,
  };
  if (record_count_ == 0u) {
    archive.hash = DiagnosticHash(archive);
    return archive;
  }
  std::vector<std::byte> bytes(static_cast<std::size_t>(byte_size_));
  archive.records.reserve(static_cast<std::size_t>(record_count_));
  std::size_t output_offset = 0u;
  for (std::uint64_t index = 0u; index < record_count_; ++index) {
    const std::uint64_t slot = Wrap(record_head_, index, records_.size());
    const Record &record = records_[static_cast<std::size_t>(slot)];
    const std::size_t size = static_cast<std::size_t>(record.byte_count);
    Copy(record.byte_offset,
         std::span<std::byte>{bytes}.subspan(output_offset, size));
    archive.records.push_back(
        ::rund::node::replay_detail::payload::DiagnosticRecord{
            .role = ::rund::node::replay_detail::payload::DiagnosticRole::
                NetworkIngress,
            .event_sequence = record.event_sequence,
            .kind = record.kind,
            .offset = static_cast<std::uint64_t>(output_offset),
            .byte_count = record.byte_count,
            .payload_hash = record.payload_hash,
        });
    output_offset += size;
  }
  archive.bytes =
      ::rund::node::replay_detail::payload::Bytes::freeze(std::move(bytes));
  archive.hash = DiagnosticHash(archive);
  return archive;
}

void RawCaptureRing::Clear() noexcept {
  byte_head_ = 0u;
  byte_size_ = 0u;
  record_head_ = 0u;
  record_count_ = 0u;
  evicted_records_ = 0u;
  dropped_records_ = 0u;
}

bool ValidDiagnosticArchive(
    const ::rund::node::replay_detail::payload::DiagnosticArchive
        &archive) noexcept {
  if (archive.records.empty()) {
    return archive.bytes.empty() && archive.report.retained_bytes == 0u &&
           archive.report.retained_records == 0u &&
           archive.hash == DiagnosticHash(archive);
  }
  if (archive.report.retained_records != archive.records.size() ||
      archive.report.retained_bytes != archive.bytes.size() ||
      archive.bytes.empty() || archive.hash == 0u) {
    return false;
  }
  std::uint64_t offset = 0u;
  std::uint64_t event_sequence = 0u;
  for (const ::rund::node::replay_detail::payload::DiagnosticRecord &record :
       archive.records) {
    if (record.role != ::rund::node::replay_detail::payload::DiagnosticRole::
                           NetworkIngress ||
        !IngressKind(record.kind) || record.event_sequence == 0u ||
        record.event_sequence <= event_sequence || record.byte_count == 0u ||
        record.offset != offset ||
        !rund::kernel::checked::add(offset, record.byte_count, offset) ||
        offset > archive.bytes.size()) {
      return false;
    }
    event_sequence = record.event_sequence;
    const std::span<const std::byte> bytes = archive.bytes.span().subspan(
        static_cast<std::size_t>(record.offset),
        static_cast<std::size_t>(record.byte_count));
    if (::rund::host::hash_bytes(bytes.data(), bytes.size()).value !=
        record.payload_hash.value) {
      return false;
    }
  }
  return offset == archive.bytes.size() &&
         DiagnosticHash(archive) == archive.hash;
}

} // namespace rund::node::replay_detail::payload
